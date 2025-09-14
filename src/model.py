import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# factory function
def create_model(config):
    model_name = config['model']['name']
    model_type = config['training'].get('model_type', 'clip')

    if model_type == 'aclip':
        keep_rate = config['training'].get('keep_rate', 0.5) # Get from config
        return ACLIP(model_name, keep_rate)
    
    return CustomCLIP(model_name)

# Attentive Masking 
def attentive_masking(image_features, text_features, keep_rate):
    """
    perform the attentive masking done by the paper at "https://arxiv.org/abs/2212.08653v2"
    Args:
        image_features (torch.Tensor): The output embeddings from the vision model.
                                       Shape: [batch_size, num_patches + 1, embed_dim]
        text_features (torch.Tensor): The output embeddings from the text model.
                                      Shape: [batch_size, embed_dim]
        keep_rate (float): The percentage of patches to keep (e.g., 0.5 for 50%).
    
    Returns:
    torch.Tensor: The selected image features.
                    Shape: [batch_size, num_kept_patches, embed_dim]
    """
     # The first token is the special [CLS] token, which we always keep.
    # The rest are the patch tokens.
    cls_token = image_features[:, :1, :]
    patch_tokens = image_features[:, 1:, :]
    
    num_patches = patch_tokens.shape[1]
    num_patches_to_keep = int(num_patches * keep_rate)

    # Normalize the features to prepare for cosine similarity calculation.
    # Normalizing is crucial for stable similarity scores.
    patch_tokens_norm = F.normalize(patch_tokens, p=2, dim=-1)
    text_features_norm = F.normalize(text_features, p=2, dim=-1)

    # Calculate the cosine similarity between each patch token and the text feature.
    # We add a dimension to text_features_norm for matrix multiplication.
    # Shape: [batch_size, num_patches]
    similarity_scores = torch.einsum('bpd,bd->bp', patch_tokens_norm, text_features_norm)

    # Find the indices of the top N patches with the highest scores.
    # `torch.topk` is perfect for this.
    # Shape: [batch_size, num_patches_to_keep]
    top_indices = torch.topk(similarity_scores, k=num_patches_to_keep, dim=-1).indices

    # Sort the indices to maintain the original spatial order (optional but good practice).
    top_indices, _ = torch.sort(top_indices, dim=-1)

    # Use `torch.gather` to select the top patches from the original patch_tokens tensor.
    # This is an efficient way to perform advanced indexing.
    # We need to expand top_indices to match the embedding dimension.
    top_indices = top_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
    kept_patch_tokens = torch.gather(patch_tokens, dim=1, index=top_indices)

    # Combine the [CLS] token we saved earlier with our selected patches.
    final_image_features = torch.cat([cls_token, kept_patch_tokens], dim=1)

    return final_image_features

# Model Definitions
class CustomCLIP(nn.Module):
    """
    wrapper for CLIP and SGLIP
    """
    def __init__(self, model_name):
        super().__init__()
        self.clip = AutoModel.from_pretrained(model_name)
    
    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.clip(
            pixel_values = pixel_values,
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_loss = True
        )
        return outputs.logits_per_image
    
class ACLIP(nn.Module):
    """An A-CLIP model that applies masking within its forward pass."""
    def __init__(self, model_name, keep_rate=0.5): # Add keep_rate here
        super().__init__()
        self.clip = AutoModel.from_pretrained(model_name)
        self.keep_rate = keep_rate
        print(f"A-CLIP model initialized. Keeping {self.keep_rate*100}% of patches.")

    def forward(self, pixel_values, input_ids, attention_mask):
        # 1. Get initial embeddings from the vision and text encoders
        # Note: We pass the features through the 'vision_model' directly
        image_outputs = self.clip.vision_model(pixel_values)
        image_features_full = image_outputs.last_hidden_state
        
        text_outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.pooler_output

        # 2. Apply A-CLIP's unique logic right here!
        kept_image_features = attentive_masking(
            image_features_full, 
            text_features, 
            self.keep_rate
        )

        # 3. Project the features and calculate the final logits
        # This part mimics the internal logic of the CLIPModel's forward pass
        image_embeds = self.clip.visual_projection(kept_image_features[:, 0, :])
        text_embeds = self.clip.text_projection(text_features)
        
        logits_per_image = self.clip.get_logits_per_image(image_embeds, text_embeds)
        
        return logits_per_image
