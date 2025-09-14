import torch
import torch.nn as nn
from typing import Dict, Any

class ClipLoss(nn.Module):
    """
    Calculates the contrastive loss for CLIP.

    This loss function computes the cross-entropy loss over similarity scores
    for a batch of image and text embeddings. It encourages the similarity
    of correct pairs to be high and incorrect pairs to be low.
    """
    def __init__(self):
        super().__init__()
        # Use PyTorch's built-in CrossEntropyLoss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits_per_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits_per_image (torch.Tensor): The similarity scores between image and text embeddings.
                                             Shape: [batch_size, batch_size]

        Returns:
            torch.Tensor: The calculated contrastive loss.
        """
        # The ground truth is that the diagonal elements (correct pairs) should be matched.
        # We create a tensor representing the indices of the correct pairs: [0, 1, 2, ...].
        batch_size = logits_per_image.shape[0]
        ground_truth = torch.arange(batch_size, device=logits_per_image.device, dtype=torch.long)

        # The loss is calculated for both image-to-text and text-to-image similarities.
        loss_img = self.loss_fn(logits_per_image, ground_truth)
        loss_txt = self.loss_fn(logits_per_image.t(), ground_truth) # Use the transpose for text-to-image

        # The final loss is the average of the two.
        return (loss_img + loss_txt) / 2.0

class SigmoidLoss(nn.Module):
    """
    Calculates the pairwise sigmoid loss for SGLIP.

    This loss treats the problem as a binary classification task for each
    image-text pair, which is more efficient for large batches.
    """
    def __init__(self):
        super().__init__()
        # Use PyTorch's binary cross-entropy loss with logits for numerical stability.
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, logits_per_image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits_per_image (torch.Tensor): The similarity scores. Shape: [batch_size, batch_size]

        Returns:
            torch.Tensor: The calculated sigmoid loss.
        """
        batch_size = logits_per_image.shape[0]
        
        # Ground truth: 1 for correct pairs (diagonal), 0 for incorrect pairs.
        ground_truth = torch.eye(batch_size, device=logits_per_image.device)

        # Calculate the binary cross-entropy loss.
        return self.loss_fn(logits_per_image, ground_truth)


def create_loss_fn(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create the appropriate loss function based on the config.
    """
    loss_type = config['training'].get('loss_type', 'clip') # Default to 'clip'

    if loss_type == 'sigmoid':
        print("Using SGLIP's Sigmoid Loss.")
        return SigmoidLoss()
    
    print("Using standard CLIP Contrastive Loss.")
    return ClipLoss()