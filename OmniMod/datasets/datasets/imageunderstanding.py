import os
import json
import random
import torch
import numpy as np
import av
from PIL import Image
from torch.utils.data import Dataset
import torchaudio

class ImageDataset(Dataset):
    def __init__(self, vis_processor, text_processor, audio_processor, image_root, ann_path, audio_dir=None):
        self.image_root = image_root
        self.audio_dir = audio_dir
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.audio_processor = audio_processor

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        info = self.ann[index]
        image_name = info['video_name']
        image_id = os.path.splitext(image_name)[0]

        image_path = os.path.join(self.image_root, image_name)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)


        answer = info['answer']
        question = info['question']
        instruction = f"<VideoHere>Please analyze step by step: {question}"

        output_data = {
            "image_id": image_id,
            "image": image, # Shape: [3, 224, 224]
            "question": question,
            "answer": answer,
            "instruction_input": instruction
        }

        # # Load audio (optional)
        # if self.audio_dir:
        #     audio_path = os.path.join(self.audio_dir, f'{image_id}.wav')
        #     if os.path.exists(audio_path):
        #         waveform, sample_rate = torchaudio.load(audio_path)
        #         # Ensure waveform is 1D (mono) and resample to 16kHz if needed
        #         if waveform.dim() > 1:
        #             waveform = waveform.mean(dim=0, keepdim=False)  # Convert to mono
        #         if sample_rate != 16000:
        #             resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        #             waveform = resampler(waveform)
        #         # Process audio with whisper_processor
        #         audio_out = self.audio_processor(waveform.numpy())
        #         audio = audio_out.squeeze()  # Should be [80, 3000]
        #         output_data["audio"] = audio
        #         # print(f"Processed audio shape: {audio.shape}")

        return output_data