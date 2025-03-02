import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from huggingface_hub import hf_hub_download
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

from base_verifier import BaseVerifier


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embed):
        return self.layers(embed)


class LAIONAestheticVerifier(BaseVerifier):
    """Based on https://github.com/christophschuhmann/improved-aesthetic-predictor."""

    SUPPORTED_METRIC_CHOICES = ["laion_aesthetic_score"]

    def __init__(self, **kwargs):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = kwargs.pop("dtype", torch.float32)

        self.clip = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
        self.clip.to(self.device, self.dtype)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.mlp = MLP()
        path = hf_hub_download("trl-lib/ddpo-aesthetic-predictor", "aesthetic-model.pth")
        state_dict = torch.load(path, weights_only=True, map_location=torch.device("cpu"))
        self.mlp.load_state_dict(state_dict)
        self.mlp.to(self.device, self.dtype)

    def prepare_inputs(self, images, prompts=None, **kwargs):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = inputs.to(device=self.device)
        inputs = {k: v.to(self.dtype) for k, v in inputs.items()}
        return inputs

    @torch.no_grad()
    @torch.inference_mode()
    def score(self, inputs, **kwargs):
        # TODO: consider batching inputs if they get too large.
        embed = self.clip(**inputs)[0]
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        scores = self.mlp(embed).squeeze(1)
        return [{"laion_aesthetic_score": score.item()} for score in scores]
