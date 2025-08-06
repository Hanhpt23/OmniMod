import torch
import torch.nn as nn
from transformers import AutoImageProcessor, VideoMAEModel

class VideoMAE(nn.Module):
    def __init__(self):
        super(VideoMAE, self).__init__()
        self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base").to(dtype=torch.float16)
        self.num_features = 768 

    def forward(self, video):
        print('video.shape', video.shape)
        return self.model(video).last_hidden_state[:, 1:, :]

def create_videomae(**kwargs):
    precision = kwargs.get("precision", "fp16")
    model = VideoMAE()
    if precision == "fp16":
        model = model.half()
    print('Using VideoMAE')
    return model

