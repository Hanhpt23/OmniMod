from transformers import CLIPModel
import torch.nn as nn

class CLIP(nn.Module):
    def __init__(self):
        super(CLIP, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model
        self.num_features = 768

    def forward(self, x):
        return self.model(x).last_hidden_state[:, 1:, :]
    
def create_clip(**kwargs):
    precision = kwargs.get("precision", "fp16")
    model = CLIP()
    if precision == "fp16":
        model = model.half()
    return model