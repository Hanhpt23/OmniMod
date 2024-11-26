import os
import json
import random
import torch
import torchaudio
from PIL import Image
from torch.utils.data import Dataset

class AudioInstruction(Dataset):
    def __init__(self, vis_processor, text_processor, audio_processor, audio_dir, vis_root, ann_path, prompt_test=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        audio_dir (string): Root direction of audio
        """
        self.vis_root = vis_root
        self.audio_dir = audio_dir

        self.vis_processor = vis_processor

        self.text_processor = text_processor
        self.audio_processor = audio_processor

        # path: ../data/LISA/json/train.json
        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        info = self.ann[index]

        image_file = info['image']
        image_id, _ = os.path.splitext(info['image'])
        image_path = os.path.join(self.vis_root, image_file)
        image = Image.open(image_path).convert("RGB")
        image = self.vis_processor(image)


        # If we have multiple outputs/queries, randomly pick one
        if isinstance(info['outputs'], list) and len(info['outputs']) > 1:
            number = random.randint(0, len(info['outputs']) - 1)  # Select a random index for query/output

            answer = info['outputs'][number]  # Select the corresponding answer
            instruction = info['query'][number]  # Select the corresponding query

            # Finding correct image and audio paths accordingly
            image_path = os.path.join(self.vis_root, f'{image_id}.jpg') 
            image = Image.open(image_path).convert("RGB")
            image = self.vis_processor(image)

            # audio
            audio_path = os.path.join(self.audio_dir, f'{image_id}_q{number + 1}.wav')  # Assuming audio file naming starts from 1
            waveform, sample_rate = torchaudio.load(audio_path)

        else:
            answer = info['outputs']  # Single answer
            instruction = info['query']  # Single query

            # audio
            audio_file, _ = os.path.splitext(info['image'])
            audio_path = os.path.join(self.audio_dir, f'{audio_file}.wav')
            waveform, sample_rate = torchaudio.load(audio_path)
            
        # # For audio instruction
        instruction = "<Img><ImageHere></Img>" 

        # # For text instruction
        # instruction = "<Img><ImageHere></Img> {} ".format(instruction)

        waveform_array = waveform.squeeze().numpy()

        waveform = self.audio_processor(waveform_array) #, sampling_rate=16000, return_tensors="pt").input_features
        waveform = waveform.squeeze()

        return {
            "image": image,
            "audio": waveform, #torch.rand(80, 3000, dtype=torch.float16), # double length of the max_source_positions
            "instruction_input": instruction,
            "answer": answer,
            "image_id": image_id,
        }
