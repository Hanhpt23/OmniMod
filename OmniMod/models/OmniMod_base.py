import logging
import random
import numpy as np
import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from OmniMod.common.registry import registry
from OmniMod.models.base_model import BaseModel
from transformers import StoppingCriteria, StoppingCriteriaList
from torch.nn import GRU  # Or small TransformerBlock for recurrence
from OmniMod.conversation.conversation import StoppingCriteriaSub

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CrossModalAttention(nn.Module):
    def __init__(self, dim=2048, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, video_emb, audio_emb):
        if audio_emb is None:
            return video_emb.unsqueeze(0), None
        if video_emb.dim() == 2:
            video_emb = video_emb.unsqueeze(0)  # [1, seq_len, hidden_dim]
        if audio_emb is not None and audio_emb.dim() == 2:
            audio_emb = audio_emb.unsqueeze(0)  # [1, seq_len, hidden_dim]
        video_emb = video_emb.transpose(0, 1)
        audio_emb = audio_emb.transpose(0, 1)
        with torch.cuda.amp.autocast(enabled=True):
            aligned_emb, attn_weights = self.attn(video_emb, audio_emb, audio_emb)
            aligned_emb = self.norm(aligned_emb + video_emb)
        aligned_emb = aligned_emb.transpose(0, 1)
        return aligned_emb, attn_weights


# class CrossModalAttention(nn.Module):
#     def __init__(self, dim=4096, reduced_dim=1024, num_heads=8):
#         super().__init__()
#         # Project to a lower-dimensional space to reduce parameters
#         self.proj_in = nn.Linear(dim, reduced_dim)
#         self.attn = nn.MultiheadAttention(embed_dim=reduced_dim, num_heads=num_heads)
#         self.norm = nn.LayerNorm(reduced_dim)
#         # Project back to original dimension if needed
#         self.proj_out = nn.Linear(reduced_dim, dim)
    
#     def forward(self, video_emb, audio_emb):
#         if audio_emb is None:
#             return video_emb.unsqueeze(0), None
#         if video_emb.dim() == 2:
#             video_emb = video_emb.unsqueeze(0)  # [1, seq_len, hidden_dim]
#         if audio_emb is not None and audio_emb.dim() == 2:
#             audio_emb = audio_emb.unsqueeze(0)  # [1, seq_len, hidden_dim]
        
#         video_emb = video_emb.transpose(0, 1)  # [seq_len, batch, hidden_dim]
#         audio_emb = audio_emb.transpose(0, 1)  # [seq_len, batch, hidden_dim]
        
#         with torch.cuda.amp.autocast(enabled=True):
#             # Project inputs to reduced dimension
#             video_emb = self.proj_in(video_emb)
#             audio_emb = self.proj_in(audio_emb)
#             # Apply attention
#             aligned_emb, attn_weights = self.attn(video_emb, audio_emb, audio_emb)
#             # Residual connection and normalization
#             aligned_emb = self.norm(aligned_emb + video_emb)
#             # Project back to original dimension
#             aligned_emb = self.proj_out(aligned_emb)
        
#         aligned_emb = aligned_emb.transpose(0, 1)  # [batch, seq_len, hidden_dim]
#         return aligned_emb, attn_weights
    
class MultimodalLatentAttention(nn.Module):
    def __init__(self, llm_hidden_dim, mixed_dim=2048, num_heads=8):
        super().__init__()
        self.proj = nn.Linear(llm_hidden_dim, mixed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=mixed_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(mixed_dim)
        self.proj_back = nn.Linear(mixed_dim, llm_hidden_dim)
        self.final_norm = nn.LayerNorm(llm_hidden_dim)
    
    def forward(self, hidden_state, mixed_embeds):
        batch_size = hidden_state.size(0)
        # Ensure hidden_state matches the dtype of proj.weight
        with torch.cuda.amp.autocast(enabled=True):
            query = self.proj(hidden_state.to(self.proj.weight.dtype)).unsqueeze(0)  # [1, batch_size, mixed_dim]
            key_value = mixed_embeds.transpose(0, 1)  # [seq_len, batch_size, mixed_dim]
            attn_output, attn_weights = self.attn(query, key_value, key_value)
            # logger.info(f"Attention weights mean: {attn_weights.mean().item()}, std: {attn_weights.std().item()}")

            attn_output = self.norm(attn_output + query)
            output = self.proj_back(attn_output.squeeze(0))  # [batch_size, hidden_dim]
            output = self.final_norm(output)  # Apply final normalization

        return output.unsqueeze(1)  # [batch_size, 1, hidden_dim]


class OmniModBase(BaseModel):
    """
    Base class for OmniModBase
    """

    def __init__(
        self,
        vision_model="eva_clip_g",
        audio_model="whisper",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        precision="fp16",
        freeze_vision=True,
        freeze_audio=True,
        language_model="",
        max_txt_len=32,
        max_context_len=3800,
        prompt_template="",
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading
        lora_r=0,  # lora_r means lora is not used
        bits=8,
        lora_target_modules=["q_proj", "v_proj"],
        lora_alpha=0.16,
        lora_dropout=0.05,
        use_coconut=False,  # Enable LatentReasoning reasoning
        use_multimodal_coconut=False,  # Enable multimodal LatentReasoning (approach 2)
        num_latent_thoughts=2,  # Number of latent thoughts (n)
        coconut_discount_rate=1.0,  # discount/encourage rate (γ = 0.9 for discounting, γ = 1.1 for encouraging)
        mu=0.3,
    ):
        super().__init__()

        self.language_model, self.language_tokenizer = self.init_llm(
            language_model_path=language_model,
            bits=bits,
            low_resource=low_resource,
            low_res_device=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # Add LatentReasoning special tokens
        self.use_coconut = use_coconut
        self.num_latent_thoughts = num_latent_thoughts
        self.use_multimodal_coconut = use_multimodal_coconut
        self.coconut_discount_rate = coconut_discount_rate  # Store discount/encourage rate
        self.mu = mu # Weight auxiliary loss

        # if self.use_coconut:
        #     self.language_tokenizer.add_special_tokens({
        #         'additional_special_tokens': ['<bot>', '<eot>']
        #     })
        #     self.language_model.resize_token_embeddings(len(self.language_tokenizer))
        #     self.bot_token_id = self.language_tokenizer.convert_tokens_to_ids('<bot>')
        #     self.eot_token_id = self.language_tokenizer.convert_tokens_to_ids('<eot>')
        #     print('self.bot_token_id, self.eot_token_id: ', self.bot_token_id, self.eot_token_id)


        self.visual_encoder, self.ln_vision, self.num_concat = self.init_vision_encoder(
            vision_model,
            freeze_vision, 
            img_size=img_size,
            drop_path_rate=drop_path_rate, 
            use_checkpoint=use_grad_checkpoint, 
            precision=precision
        )

        self.audio_encoder = self.init_audio_encoder(
            audio_model,
            freeze_audio,
            precision=precision
        )
        DIM = self.language_model.config.hidden_size # 2048 for 3.2 1B | 4096 for Llama-3.1-8B-Instruct
        # Initialize attention-based mixer for video and audio embedding
        self.cross_modal_attn = CrossModalAttention(dim=DIM, num_heads=8)
        
        # Initialize multimodal latent attention for approach 2
        if self.use_multimodal_coconut:
            self.llm_hidden_dim = self.language_model.config.hidden_size 
            self.multimodal_latent_attn = MultimodalLatentAttention(
                llm_hidden_dim=self.llm_hidden_dim, mixed_dim=DIM, num_heads=8)

        self.precision = precision
        self.max_txt_len = max_txt_len
        self.max_context_len = max_context_len
        self.end_sym = end_sym
        self.prompt_template = prompt_template
        self.prompt_list = []


    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()


    def mix_video_audio(self, video_embeds, audio_embeds=None):
        """Mix video and audio embeddings using multi-head attention."""
        if audio_embeds is None:
            if video_embeds.dim() == 2:
                video_embeds = video_embeds.unsqueeze(0)
            return video_embeds
        # Ensure batch dimension
        if video_embeds.dim() == 2:
            video_embeds = video_embeds.unsqueeze(0)  # [1, seq_len, hidden_dim]
        if audio_embeds.dim() == 2:
            audio_embeds = audio_embeds.unsqueeze(0)
        # Apply cross-modal attention
        aligned_emb, _ = self.cross_modal_attn(video_embeds, audio_embeds)
        return aligned_emb

    def get_context_emb(self, prompt, video_list, audio=None):
        device = video_list[0].device
        prompt_segs = prompt.split('<VideoHere>')
        assert len(prompt_segs) == len(video_list) + 1, "Unmatched numbers of video placeholders and videos."
        seg_tokens = [
            self.language_tokenizer(seg, return_tensors="pt", add_special_tokens=False).to(device).input_ids
            for seg in prompt_segs
        ]
        with torch.cuda.amp.autocast(enabled=self.precision == "fp16"):
            seg_embs = [self.embed_tokens(seg_t) for seg_t in seg_tokens]
            mixed_embs = [
                self.mix_video_audio(video_emb, audio[i] if audio is not None else None)
                for i, video_emb in enumerate(video_list)
            ]
        interleaved_embs = [emb for pair in zip(seg_embs[:-1], mixed_embs) for emb in pair] + [seg_embs[-1]]
        return torch.cat(interleaved_embs, dim=1), mixed_embs
    
    def pad_mixed_embeds(self, mixed_embs_list):
        """Pad mixed embeddings to the maximum sequence length in the batch."""
        # Flatten nested lists if necessary
        flat_mixed_embs = []
        for emb in mixed_embs_list:
            if isinstance(emb, list):
                flat_mixed_embs.extend(emb)
            else:
                flat_mixed_embs.append(emb)
        
        # Verify all elements are tensors
        for i, emb in enumerate(flat_mixed_embs):
            if not isinstance(emb, torch.Tensor):
                raise ValueError(f"Element {i} in mixed_embs_list is not a tensor: {type(emb)}")
        
        max_seq_len = max(emb.shape[1] for emb in flat_mixed_embs)
        batch_size = len(flat_mixed_embs)
        device = flat_mixed_embs[0].device
        mixed_dim = flat_mixed_embs[0].shape[2]
        
        # logger.info(f"Padding mixed_embs: batch_size={batch_size}, max_seq_len={max_seq_len}, mixed_dim={mixed_dim}")

        padded_embs = torch.zeros(batch_size, max_seq_len, mixed_dim, dtype=flat_mixed_embs[0].dtype, device=device)
        for i, emb in enumerate(flat_mixed_embs):
            seq_len = emb.shape[1]
            padded_embs[i, :seq_len, :] = emb
        return padded_embs
    
    def prompt_wrap(self, video_embeds, audio_embeds, video_atts, prompts, lengths=None):
        """Wrap prompts with mixed video and audio embeddings for video VQA."""
        emb_lists = []
        mixed_embs_list = []
        audio_iter = audio_embeds if audio_embeds is not None else [None] * len(video_embeds)
        for idx, (v_emb, prompt, a_emb) in enumerate(zip(video_embeds, prompts, audio_iter)):
            pn = v_emb.shape[-2]
            if lengths is not None:
                v_emb = v_emb.reshape(-1, v_emb.shape[-1])[:lengths[idx] * pn]
            mixed_emb = self.mix_video_audio(v_emb, a_emb)
            p_segs = prompt.split('<VideoHere>')
            interleave_emb = []
            for inner_idx, seg in enumerate(p_segs[:-1]):
                p_tokens = self.language_tokenizer(seg, return_tensors="pt", add_special_tokens=False).to(video_embeds.device)
                p_embed = self.embed_tokens(p_tokens.input_ids)
                segment_length = pn if lengths is None else lengths[idx]
                start = inner_idx * segment_length
                end = (inner_idx + 1) * segment_length
                mixed_segment = mixed_emb[:, start:end] if start < mixed_emb.shape[1] else mixed_emb[:, -pn:]
                interleave_emb.append(torch.cat([p_embed, mixed_segment], dim=1))
            wrapped_emb = torch.cat(interleave_emb, dim=1)
            p_tokens = self.language_tokenizer(p_segs[-1], return_tensors="pt", add_special_tokens=False).to(video_embeds.device)
            p_embed = self.embed_tokens(p_tokens.input_ids)
            wrapped_emb = torch.cat([wrapped_emb, p_embed], dim=1)
            emb_lists.append(wrapped_emb)
            mixed_embs_list.append(mixed_emb)
        # print('prompt_wrap - mixed_embs_list: ', len(mixed_embs_list))

        emb_lens = [emb.shape[1] for emb in emb_lists]
        pad_emb = self.embed_tokens(torch.tensor(self.language_tokenizer.pad_token_id, device=video_embeds.device))
        max_length = min(max(emb_lens), self.max_context_len)
        wrapped_embs = pad_emb.expand(len(emb_lens), max_length, -1).clone()
        wrapped_atts = torch.zeros([len(emb_lens), max_length], dtype=torch.int, device=video_embeds.device)
        for i, emb in enumerate(emb_lists):
            length = min(emb_lens[i], self.max_context_len)
            wrapped_embs[i, :length] = emb[:, :length]
            wrapped_atts[i, :length] = 1

        # Pad mixed_embs_list for multimodal LatentReasoning
        padded_mixed_embs = self.pad_mixed_embeds(mixed_embs_list)
        # padded_mixed_embs = self.pad_mixed_embeds(mixed_embs_list, len(prompts))
        return wrapped_embs, wrapped_atts, padded_mixed_embs

    def concat_emb_input_output(self, input_embs, input_atts, output_embs, output_atts):
        input_lens = []
        cat_embs = []
        cat_atts = []
        for i in range(input_embs.size(0)):
            input_len = input_atts[i].sum()
            input_lens.append(input_len)
            cat_embs.append(
                torch.cat([
                    input_embs[i][:input_len],
                    output_embs[i],
                    input_embs[i][input_len:]
                ])
            )
            cat_atts.append(
                torch.cat([
                    input_atts[i][:input_len],
                    output_atts[i],
                    input_atts[i][input_len:]
                ])
            )
        cat_embs = torch.stack(cat_embs)
        cat_atts = torch.stack(cat_atts)
        return cat_embs, cat_atts, input_lens

    def tokenize_conversation(self, conv_q, conv_a):
        to_regress_token_ids_list = []
        targets_list = []
        batch_size = len(conv_q)
        for batch_idx in range(batch_size):
            questions, answers = conv_q[batch_idx], conv_a[batch_idx]
            questions = [self.language_tokenizer(self.language_tokenizer.bos_token + q,
                                              return_tensors="pt",
                                              add_special_tokens=False).to(self.device) for q in questions[1:]]
            answers = [self.language_tokenizer(a + self.end_sym,
                                            return_tensors="pt",
                                            add_special_tokens=False).to(self.device) for a in answers]
            cur_id = []
            cur_target = []
            for i in range(len(questions)):
                cur_id.append(answers[i].input_ids)
                cur_target.append(answers[i].input_ids)
                cur_id.append(questions[i].input_ids)
                cur_target.append(torch.ones_like(questions[i].input_ids) * -100)
            cur_id.append(answers[-1].input_ids)
            cur_target.append(answers[-1].input_ids)
            cur_id = torch.cat(cur_id, dim=1)
            cur_target = torch.cat(cur_target, dim=1)
            to_regress_token_ids_list.append(cur_id)
            targets_list.append(cur_target)
        max_len = min(max([target.shape[1] for target in targets_list]), self.max_txt_len)
        to_regress_token_ids = torch.ones([batch_size, max_len],
                                          dtype=cur_id.dtype, device=self.device) * self.language_tokenizer.pad_token_id
        targets = torch.ones([batch_size, max_len],
                             dtype=cur_id.dtype, device=self.device) * -100
        for batch_idx in range(batch_size):
            cur_len = to_regress_token_ids_list[batch_idx].shape[1]
            to_regress_token_ids[batch_idx, :cur_len] = to_regress_token_ids_list[batch_idx][0, :max_len]
            targets[batch_idx, :cur_len] = targets_list[batch_idx][0, :max_len]
        to_regress_token_attn = (to_regress_token_ids != self.language_tokenizer.pad_token_id).to(torch.int)
        return to_regress_token_ids, to_regress_token_attn, targets

    def preparing_embedding(self, samples):
        if "audio" in samples:
            audio_embeds, audio_atts = self.encode_audio(samples["audio"])
        else:
            audio_embeds, audio_atts = None, None
        if 'image' in samples:
            img_embeds, img_atts = self.encode_img(samples["image"])
        else:
            img_embeds = img_atts = None
        if 'conv_q' in samples:
            conv_q, conv_a = samples['conv_q'], samples['conv_a']
            connect_sym = samples['connect_sym'][0]
            conv_q = [q.split(connect_sym) for q in conv_q]
            conv_a = [a.split(connect_sym) for a in conv_a]
            conv_q = [[self.prompt_template.format(item) for item in items] for items in conv_q]
            cond_embeds, cond_atts, padded_mixed_embs = self.prompt_wrap(img_embeds, audio_embeds, img_atts, [q[0] for q in conv_q])
            regress_token_ids, regress_atts, part_targets = self.tokenize_conversation(conv_q, conv_a)
        else:
            if "instruction_input" in samples:
                instruction = samples["instruction_input"]
            elif self.prompt_list:
                instruction = random.choice(self.prompt_list)
            else:
                instruction = None
            if hasattr(self, 'chat_template') and self.chat_template:
                instruction = [self.prompt_template.format(instruct) for instruct in instruction]
            if 'length' in samples:
                cond_embeds, cond_atts, padded_mixed_embs = self.prompt_wrap(img_embeds, audio_embeds, img_atts, instruction, samples['length'])
            else:
                cond_embeds, cond_atts, padded_mixed_embs = self.prompt_wrap(img_embeds, audio_embeds, img_atts, instruction)

            ### prepare target tokens
            self.language_tokenizer.padding_side = "right"
            text = [t + self.end_sym for t in samples["answer"]]

            regress_tokens = self.language_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(self.device)

            regress_token_ids = regress_tokens.input_ids
            regress_atts = regress_tokens.attention_mask
            part_targets = regress_token_ids.masked_fill(
                regress_token_ids == self.language_tokenizer.pad_token_id, -100
            )

        regress_embeds = self.embed_tokens(regress_token_ids)
        return cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets, padded_mixed_embs

    def get_last_hidden_state(self, inputs_embeds, attention_mask):
        """Extract the last hidden state from the LLM."""
        with self.maybe_autocast():
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True
            )
        # Get the last hidden state (batch_size, seq_len, hidden_dim)
        last_hidden_state = outputs.hidden_states[-1]
        # Select the last non-padded token's hidden state for each sample
        batch_size, seq_len, hidden_dim = last_hidden_state.shape
        last_positions = attention_mask.sum(dim=1) - 1  # Index of last non-padded token
        last_hidden = torch.gather(
            last_hidden_state,
            dim=1,
            index=last_positions.unsqueeze(1).unsqueeze(2).expand(-1, 1, hidden_dim)
        ).squeeze(1)  # [batch_size, hidden_dim]
        return last_hidden

    def forward(self, samples, reduction='mean'):
        """Run the training forward pass with optional LatentReasoning latent reasoning."""
        cond_embeds, cond_atts, regress_embeds, regress_atts, part_targets, padded_mixed_embs = \
            self.preparing_embedding(samples)
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)
        bos = torch.ones_like(part_targets[:, :1]) * self.language_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = cond_atts[:, :1]
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)
        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)
        for i, target in enumerate(part_targets):
            targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target  # plus 1 for bos

        if not self.use_coconut:
            # print('forward: Using standard training')
            # Standard training (single forward pass)
            with self.maybe_autocast():
                outputs = self.language_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                    reduction=reduction
                )
            return {"loss": outputs.loss}

        else:
            batch_size, seq_len, emb_dim = inputs_embeds.shape
            latent_embeds = inputs_embeds
            latent_attention_mask = attention_mask
            latent_targets = targets.clone()
            total_loss = 0.0

            # bot_embeds = self.embed_tokens(
            #     torch.ones([batch_size, 1], dtype=torch.long, device=self.device) * self.bot_token_id
            # )
            # eot_embeds = self.embed_tokens(
            #     torch.ones([batch_size, 1], dtype=torch.long, device=self.device) * self.eot_token_id
            # )

            for thought_idx in range(self.num_latent_thoughts + 1):
                with self.maybe_autocast():
                    if thought_idx < self.num_latent_thoughts:
                        # Latent mode: get last hidden state and append <bot> token
                        last_hidden = self.get_last_hidden_state(latent_embeds, latent_attention_mask)

                        # Apply discount/encourage rate: γ^thought_idx
                        gamma = self.coconut_discount_rate ** thought_idx
                        last_hidden = last_hidden * gamma

                        if self.use_multimodal_coconut:
                            # print('Forward - LatentReasoning + MultiMix')
                            # Combine hidden state with mixed embeddings
                            thought_emb = self.multimodal_latent_attn(last_hidden, padded_mixed_embs)
                        else:
                            # print('Forward - LatentReasoning')
                            thought_emb = last_hidden.unsqueeze(1)  # [batch_size, 1, hidden_dim]
                        # latent_embeds = torch.cat([latent_embeds, bot_embeds, thought_emb, eot_embeds], dim=1)
                        latent_embeds = torch.cat([latent_embeds, thought_emb], dim=1)
                        # latent_embeds = thought_emb
                        latent_attention_mask = torch.cat(
                            [latent_attention_mask, torch.ones([batch_size, 1], dtype=torch.int, device=self.device)],
                            dim=1
                        )
                        latent_targets = torch.cat(
                            [latent_targets, torch.ones([batch_size, 1], dtype=torch.long, device=self.device) * -100],
                            dim=1
                        )

                        if self.mu:
                            # Add auxiliary loss for intermediate thoughts
                            aux_outputs = self.language_model(inputs_embeds=latent_embeds, attention_mask=latent_attention_mask, return_dict=True, labels=latent_targets)
                            # logger.info(f"aux_outputs: {aux_outputs.loss}")
                            total_loss += aux_outputs.loss * self.mu #0.3  # Weight auxiliary loss

                    else:
                        # Final language mode: compute loss
                        outputs = self.language_model(
                            inputs_embeds=latent_embeds,
                            attention_mask=latent_attention_mask,
                            return_dict=True,
                            labels=latent_targets,
                            reduction=reduction
                        )
                        total_loss += outputs.loss
            # return {"loss": total_loss / (self.num_latent_thoughts + 1)}
            return {"loss": total_loss}

    def embed_tokens(self, token_ids):
        if hasattr(self.language_model.base_model, 'model'):
            embeds = self.language_model.base_model.model.model.embed_tokens(token_ids)
        else:
            embeds = self.language_model.base_model.embed_tokens(token_ids)
        return embeds

    def generate(
        self,
        images=None,
        audios=None,
        texts=None,
        num_beams=1,
        max_new_tokens=20,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1,
        length_penalty=1,
        temperature=1,
        do_sample=False,
        stop_words_ids=[2],
    ):
        if images is not None and texts is None:
            raise ValueError("You must specify <VideoHere> in the text")
        stopping_criteria = StoppingCriteriaList([
            StoppingCriteriaSub(stops=[torch.tensor([i]).to(self.device) for i in stop_words_ids])
        ])

        if images is not None:
            if isinstance(images, list):
                images = np.array(images, dtype=np.float16)
                images = torch.from_numpy(images).to(self.device)
            else:
                images = images.to(self.device)
        if audios is not None:
            if isinstance(audios, list):
                audios = np.array(audios, dtype=np.float16)
                audios = torch.from_numpy(audios).to(self.device)
            else:
                audios = audios.to(self.device)
        with torch.cuda.amp.autocast(enabled=self.precision == "fp16"):
            img_embeds, atts_img = self.encode_img(images) if images is not None else (None, None)
            audio_embeds = [self.encode_audio(audios)[0] if audios is not None else None] * (len(texts) if texts else 1)
            image_lists = [[image_emb[None]] for image_emb in img_embeds] if img_embeds is not None else None
            batch_embs = [
                self.get_context_emb(text, img_list, audio_emb if audios is not None else None)
                for text, img_list, audio_emb in zip(texts or [""], image_lists or [None], audio_embeds)
            ]
        batch_size = len(batch_embs)

      # Flatten mixed_embs_list to handle nested lists
        mixed_embs_list = []
        for emb in batch_embs:
            mixed_embs = emb[1]
            if isinstance(mixed_embs, list):
                mixed_embs_list.extend(mixed_embs)
            else:
                mixed_embs_list.append(mixed_embs)

        batch_embs = [emb[0] for emb in batch_embs]  # Extract context embeddings
        max_len = max([emb.shape[1] for emb in batch_embs])
        emb_dim = batch_embs[0].shape[2]
        dtype = batch_embs[0].dtype
        device = batch_embs[0].device
        embs = torch.zeros([batch_size, max_len, emb_dim], dtype=dtype, device=device)
        attn_mask = torch.zeros([batch_size, max_len], dtype=torch.int, device=device)
        for i, emb in enumerate(batch_embs):
            emb_len = emb.shape[1]
            embs[i, -emb_len:] = emb[0]
            attn_mask[i, -emb_len:] = 1


        bos = torch.ones([batch_size, 1], dtype=torch.long, device=device) * self.language_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        embs = torch.cat([bos_embeds, embs], dim=1)
        attn_mask = torch.cat([torch.ones([batch_size, 1], dtype=torch.int, device=device), attn_mask], dim=1)

        if not self.use_coconut:
            # print('Generate: Using standard training')
            # Standard generation
            with torch.cuda.amp.autocast(enabled=self.precision == "fp16"):
                outputs = self.language_model.generate(
                    inputs_embeds=embs,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    temperature=temperature,
                    do_sample=do_sample,
                    min_length=min_length,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    stopping_criteria=stopping_criteria
                )


        else:
            # LatentReasoning generation with latent thoughts
            latent_embeds = embs
            latent_attn_mask = attn_mask
            # print('Generate - mixed_embs_list: ', len(mixed_embs_list))
            padded_mixed_embs = self.pad_mixed_embeds(mixed_embs_list)
            # padded_mixed_embs = self.pad_mixed_embeds(mixed_embs_list).to(dtype=self.multimodal_latent_attn.proj.weight.dtype)
            # bot_embeds = self.embed_tokens(
            #     torch.ones([batch_size, 1], dtype=torch.long, device=device) * self.bot_token_id
            # )
            # eot_embeds = self.embed_tokens(
            #     torch.ones([batch_size, 1], dtype=torch.long, device=device) * self.eot_token_id
            # )

            for thought_idx in range(self.num_latent_thoughts):
                last_hidden = self.get_last_hidden_state(latent_embeds, latent_attn_mask)
                gamma = self.coconut_discount_rate ** thought_idx
                last_hidden = last_hidden * gamma

                if self.use_multimodal_coconut:
                    # print('Generate - LatentReasoning + MultiMix')
                    # Combine hidden state with mixed embeddings
                    thought_emb = self.multimodal_latent_attn(last_hidden, padded_mixed_embs)
                else:
                    # print('Generate - LatentReasoning')
                    thought_emb = last_hidden.unsqueeze(1)
                latent_embeds = torch.cat([latent_embeds, thought_emb], dim=1)
                latent_attn_mask = torch.cat(
                    [latent_attn_mask, torch.ones([batch_size, 1], dtype=torch.int, device=device)],
                    dim=1
                )

            with torch.cuda.amp.autocast(enabled=self.precision == "fp16"):
                outputs = self.language_model.generate(
                    inputs_embeds=latent_embeds,
                    attention_mask=latent_attn_mask,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    temperature=temperature,
                    do_sample=do_sample,
                    min_length=min_length,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    stopping_criteria=stopping_criteria
                )

        answers = []
        for output_token in outputs:
            if output_token[0] == 0:
                output_token = output_token[1:]
            output_texts = self.language_tokenizer.decode(output_token, skip_special_tokens=True)
            output_texts = output_texts.split('</s>')[0]
            output_texts = output_texts.replace("<s>", "")
            output_texts = output_texts.split(r'[/INST]')[-1].strip()
            answers.append(output_texts)
        return answers
    