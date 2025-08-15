from OmniMod.common.registry import registry
from OmniMod.processors.base_processor import BaseProcessor
from omegaconf import OmegaConf

from transformers import AutoImageProcessor, VideoMAEModel

@registry.register_processor("videomae_processor")
class VideoProcessor(BaseProcessor):
    def __init__(self, model_name="MCG-NJU/videomae-base"):
        self.video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

    def __call__(self, video):
        # Process the waveform using the WhisperProcessor
        return self.video_processor(video, return_tensors="pt")

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        model_name = cfg.get("model_name", "MCG-NJU/videomae-base")

        return cls(model_name=model_name)
