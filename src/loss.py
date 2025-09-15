import torch
import torch.nn as nn
from typing import Dict, Any

class ClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits_per_image: torch.Tensor) -> torch.Tensor:
        batch_size = logits_per_image.shape[0]
        ground_truth = torch.arange(batch_size, device=logits_per_image.device, dtype=torch.long)
        loss_img = self.loss_fn(logits_per_image, ground_truth)
        loss_txt = self.loss_fn(logits_per_image.t(), ground_truth)
        return (loss_img + loss_txt) / 2.0

class SigmoidLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits_per_image: torch.Tensor) -> torch.Tensor:
        batch_size = logits_per_image.shape[0]
        ground_truth = torch.eye(batch_size, device=logits_per_image.device)
        return self.loss_fn(logits_per_image, ground_truth)

def create_loss_fn(config: Dict[str, Any]) -> nn.Module:
    loss_type = config['training'].get('loss_type', 'clip')

    if loss_type == 'sigmoid':
        print("Using Sigmoid Loss for SigLIP.")
        return SigmoidLoss()
    
    print("Using standard CLIP Contrastive Loss.")
    return ClipLoss()