import torch.nn as nn
from transformers import AutoModel
from typing import Dict, Any

class CustomCLIP(nn.Module):
    """A simple wrapper for the standard OpenAI CLIP model."""
    def __init__(self, model_name: str):
        super().__init__()
        self.clip = AutoModel.from_pretrained(model_name)

    def forward(self, pixel_values, input_ids, attention_mask, **kwargs):
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.logits_per_image

class SiglipForFineTuning(nn.Module):
    """A wrapper for the official Google SigLIP model."""
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, pixel_values, input_ids, **kwargs):
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids
        )
        return outputs.logits_per_image

def create_model(config: Dict[str, Any]) -> nn.Module:
    """Factory function to create the correct model based on the config."""
    model_name = config['model']['name']
    model_type = config['model'].get('type', 'clip')

    if model_type == 'siglip':
        print(f"Loading official SigLIP model: {model_name}")
        return SiglipForFineTuning(model_name)
    
    print(f"Loading standard CLIP model: {model_name}")
    return CustomCLIP(model_name)