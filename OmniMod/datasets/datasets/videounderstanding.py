import os
import json
import random
import torch
import numpy as np
import av
from PIL import Image
from torch.utils.data import Dataset
import torchaudio

class VideoDataset(Dataset):
    def __init__(self, vis_processor, text_processor, audio_processor, video_root, ann_path, audio_dir=None):
        self.video_root = video_root
        self.audio_dir = audio_dir
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.audio_processor = audio_processor

        with open(ann_path, 'r') as f:
            self.ann = json.load(f)

    def __len__(self):
        return len(self.ann)

    def read_video_pyav(self, video_path, indices):
        container = av.open(video_path)
        frames = []
        container.seek(0)
        start_index = indices[0]
        end_index = indices[-1]
        for i, frame in enumerate(container.decode(video=0)):
            if i > end_index:
                break
            if i in indices:
                frames.append(frame.to_ndarray(format="rgb24"))
        if not frames:
            raise RuntimeError(f"No frames extracted from {video_path} for indices {indices}")
        return np.stack(frames)

    def sample_frame_indices(self, clip_len, frame_sample_rate, seg_len):
        converted_len = int(clip_len * frame_sample_rate)
        if converted_len >= seg_len:
            start_idx = 0
            end_idx = seg_len - 1
        else:
            end_idx = np.random.randint(converted_len, seg_len)
            start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        return indices

    def preprocess_video_clip_videomae(self, video_path, clip_len=16):
        try:
            container = av.open(video_path)
            seg_len = container.streams.video[0].frames
            # Check if video has valid frames
            if seg_len == 0:
                raise RuntimeError(f"Video {video_path} has no frames")
            # Dynamically calculate frame_sample_rate as seg_len // clip_len
            frame_sample_rate = max(1, seg_len // clip_len)
            # If video is shorter than clip_len, use all frames and pad
            if seg_len < clip_len:
                indices = list(range(seg_len)) + [seg_len - 1] * (clip_len - seg_len)
            else:
                indices = self.sample_frame_indices(clip_len, frame_sample_rate, seg_len)
            video = self.read_video_pyav(video_path, indices)
            images = [Image.fromarray(frame) for frame in video]
            inputs = self.vis_processor(images)
            return inputs["pixel_values"]
        except Exception as e:
            raise RuntimeError(f"Failed to process {video_path}: {e}")

    def __getitem__(self, index):

        info = self.ann[index]
        video_name = info['video_name']
        video_id = os.path.splitext(video_name)[0]

        video_path = os.path.join(self.video_root, video_name)

        video = self.preprocess_video_clip_videomae(video_path, clip_len=16)

        answer = info['answer']
        question = info['question']
        instruction = f"<VideoHere>Please analyze step by step: {question}"

        output_data = {
            "image_id": video_id,
            "image": video.squeeze(0), # Shape: [16, 3, 224, 224]
            "question": question,
            "answer": answer,
            "instruction_input": instruction
        }

        # # Load audio (optional)
        # if self.audio_dir:
        #     audio_path = os.path.join(self.audio_dir, f'{video_id}.wav')
        #     if os.path.exists(audio_path):
        #         waveform, sample_rate = torchaudio.load(audio_path)
        #         audio_out = self.audio_processor(waveform.squeeze().numpy())
        #         audio = audio_out.squeeze()
        #         output_data["audio"] = audio

        # print('video.squeeze(0): ', video.squeeze(0).shape, audio.shape)

        # return output_data


        # Load audio (optional)
        if self.audio_dir:
            audio_path = os.path.join(self.audio_dir, f'{video_id}.wav')
            if os.path.exists(audio_path):
                waveform, sample_rate = torchaudio.load(audio_path)
                # Ensure waveform is 1D (mono) and resample to 16kHz if needed
                if waveform.dim() > 1:
                    waveform = waveform.mean(dim=0, keepdim=False)  # Convert to mono
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                    waveform = resampler(waveform)
                # Process audio with whisper_processor
                audio_out = self.audio_processor(waveform.numpy())
                audio = audio_out.squeeze()  # Should be [80, 3000]
                output_data["audio"] = audio
                # print(f"Processed audio shape: {audio.shape}")

        return output_data