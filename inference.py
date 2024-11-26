import json
from OmniMod.common.registry import registry
from OmniMod.common.config import Config
from OmniMod.conversation.conversation import Conversation, SeparatorStyle
from PIL import Image
import torchaudio
import argparse

# Define conversation template
CONV_VISION = Conversation(
    system="",
    roles=(r"<s>[INST] ", r" [/INST]"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="",
)

def prepare_texts(texts, conv_temp):
    convs = [conv_temp.copy() for _ in range(len(texts))]
    [conv.append_message(
        conv.roles[0], '{}'.format(text)) for conv, text in zip(convs, texts)]
    [conv.append_message(conv.roles[1], None) for conv in convs]
    texts = [conv.get_prompt() for conv in convs]
    return texts

def load_model(config_path):
    cfg = Config(config_path)
    model_config = cfg.model_cfg
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:0')
    model.eval()

    # Load processors
    key = list(cfg.datasets_cfg.keys())[0]
    vis_processor_cfg = cfg.datasets_cfg.get(key).vis_processor.train
    text_processor_cfg = cfg.datasets_cfg.get(key).text_processor.train
    audio_processor_cfg = cfg.datasets_cfg.get(key).audio_processor.train

    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    text_processor = registry.get_processor_class(text_processor_cfg.name).from_config(text_processor_cfg)
    audio_processor = registry.get_processor_class(audio_processor_cfg.name).from_config(audio_processor_cfg)

    return model, vis_processor, text_processor, audio_processor

def generate_from_inputs(model, vis_processor, audio_processor, input_list, max_new_tokens=50, temperature=1.0, top_p=0.9, do_sample=False):
    conv_temp = CONV_VISION.copy()
    results = []

    for item in input_list:
        audio_path = item["audio"]
        image_path = item["image"]
        text = item["text"]

        image = Image.open(image_path).convert('RGB')
        image = vis_processor(image)
        waveform, _ = torchaudio.load(audio_path)
        waveform_array = waveform.squeeze().numpy()
        waveform = audio_processor(waveform_array)

        texts = prepare_texts([text], conv_temp)
        predicts = model.generate(images=image,
                                  audios=waveform,
                                  texts=texts,
                                  max_new_tokens=max_new_tokens,
                                  temperature=temperature,
                                  top_p=top_p,
                                  do_sample=do_sample)
        results.append({"input": text, "prediction": predicts[0]})

    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="path_to_your_config_file.yaml")
    parser.add_argument("--audio", type=str, default="path_to_your_audio_file.wav")
    parser.add_argument("--image", type=str, default="path_to_your_image_file.jpg")
    parser.add_argument("--text", type=str, default="path_to_your_image_file.jpg")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    config_path = args.config_path
    model, vis_processor, text_processor, audio_processor = load_model(config_path)
    
    input_list = [
        {"audio": args.audio, "image": args.image, "text": args.text},
    ]

    predictions = generate_from_inputs(model, vis_processor, audio_processor, input_list)
    print(json.dumps(predictions, indent=2, ensure_ascii=False))
