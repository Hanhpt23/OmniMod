import os
import argparse
import json
import logging
from OmniMod.datasets.datasets.videounderstanding import VideoDataset
from OmniMod.datasets.datasets.imageunderstanding import ImageDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from OmniMod.common.registry import registry
from OmniMod.common.config import Config
from OmniMod.conversation.conversation import Conversation, SeparatorStyle
import torch

torch.cuda.empty_cache()

CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

def list_of_str(arg):
    return list(map(str, arg.split(',')))

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--cfg-path", required=True, help="path to evaluate configuration file.")
    parser.add_argument("--eval-dataset", type=list_of_str, default='video_val', help="dataset to evaluate")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file.",
    )
    return parser.parse_args()

def prepare_texts(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(conv.roles[0], '{}'.format(text)) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts

def init_model(cfg):
    logging.info('Initializing Model')
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')
    key = list(cfg.datasets_cfg.keys())[0]
    vis_processor_cfg = cfg.datasets_cfg.get(key).vis_processor.train
    text_processor_cfg = cfg.datasets_cfg.get(key).text_processor.train
    audio_processor_cfg = cfg.datasets_cfg.get(key).audio_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)
    audio_processor = registry.get_processor_class(audio_processor_cfg.name).from_config(audio_processor_cfg)
    logging.info('Initialization Finished')
    return model, vis_processor, text_processor, audio_processor, vis_processor_cfg.name

def evaluate(args):
    cfg = Config(args)
    model, vis_processor, text_processor, audio_processor, vis_processor_name = init_model(cfg)
    model.eval()
    conv_temp = CONV_VISION.copy()

    for dataset in args.eval_dataset:
        eval_file_path = cfg.evaluation_datasets_cfg[dataset]["ann_path"]
        img_path = cfg.evaluation_datasets_cfg[dataset]["image_path"]
        prompt_test = cfg.evaluation_datasets_cfg[dataset]["prompt_test"]
        batch_size = cfg.evaluation_datasets_cfg[dataset]["batch_size"]
        max_new_tokens = cfg.evaluation_datasets_cfg[dataset]["max_new_tokens"]
        temperature = cfg.evaluation_datasets_cfg[dataset]["temperature"]
        top_p = cfg.evaluation_datasets_cfg[dataset]["top_p"]
        do_sample = cfg.evaluation_datasets_cfg[dataset]["do_sample"]
        audio_path = cfg.evaluation_datasets_cfg[dataset]["audio_path"]

        if vis_processor_name == 'blip2_image_eval':
            data = ImageDataset(
                vis_processor=vis_processor,
                text_processor=text_processor,
                audio_processor=audio_processor,
                audio_dir=audio_path,
                ann_path=eval_file_path,
                image_root=img_path
            )
        elif vis_processor_name == 'videomae_processor':
            data = VideoDataset(
                vis_processor=vis_processor,
                text_processor=text_processor,
                audio_processor=audio_processor,
                audio_dir=audio_path,
                ann_path=eval_file_path,
                video_root=img_path
            )
        else:
            raise RuntimeError(f"Can not find suitable vision processor for the vision input type!")
        
        eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        results = []
        for batch in tqdm(eval_dataloader):
            # logging.info(f"evaluate: batch keys: {list(batch.keys())}, audio present: {'audio' in batch}")
            images = batch["image"].half()
            audios = batch.get("audio", None)
            instruction_input = batch["instruction_input"]
            ground_truth = batch["answer"]
            text_questions = batch["question"]
            image_ids = batch["image_id"]
            texts = prepare_texts(instruction_input, conv_temp)
            predicts = model.generate(images=images,
                                      audios=audios,
                                      texts=texts,
                                      max_new_tokens=max_new_tokens,
                                      temperature=temperature,
                                      top_p=top_p,
                                      do_sample=do_sample)

            results.extend([{
                "image_id": image_id,
                "text_question": text_question,
                "ground_truth": gt,
                "predict": predict
            } for image_id, text_question, gt, predict in zip(image_ids, text_questions, ground_truth, predicts)])
            logging.info(f"evaluate: predicts: {predicts[:2]}")
            break

        ckpt_path, ckpt_name = os.path.split(cfg.model_cfg.ckpt)
        save_path = os.path.join(ckpt_path, 'result', f"output_{ckpt_name.split('.')[0]}.json")
        with open(save_path, "w") as jsonfile:
            json.dump(results, jsonfile, ensure_ascii=False)
        logging.info(f'Saving results to: {save_path}')

if __name__ == "__main__":
    args = parse_args()
    logging.info("Evaluating...")
    evaluate(args)
    logging.info("Done!")