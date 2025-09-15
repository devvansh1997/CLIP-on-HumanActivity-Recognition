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
        # 1. Get raw outputs from both encoders
        image_outputs = self.clip.vision_model(pixel_values)
        image_features_raw = image_outputs.last_hidden_state # Shape: [batch, patches, 768]

        text_outputs = self.clip.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features_raw = text_outputs.last_hidden_state # Shape: [batch, tokens, 512]

        # 2. --- NEW: Project features into the shared latent space FIRST ---
        # The visual_projection can handle the sequence of patches
        projected_image_features = self.clip.visual_projection(image_features_raw)
        
        # We use the [CLS] token's output for the text representation
        pooled_text_output = text_outputs.pooler_output
        projected_text_features = self.clip.text_projection(pooled_text_output)

        # 3. Apply attentive masking using the CORRECTLY SIZED projected features
        kept_image_features = attentive_masking(
            projected_image_features, 
            projected_text_features, 
            self.keep_rate
        )

        # 4. Normalize the final embeddings and calculate logits
        # We take the [CLS] token from the kept image features
        final_image_embeds = kept_image_features[:, 0, :]
        
        final_image_embeds = F.normalize(final_image_embeds, p=2, dim=-1)
        final_text_embeds = F.normalize(projected_text_features, p=2, dim=-1)

        # Calculate logits using the model's logit scale
        logit_scale = self.clip.logit_scale.exp()
        logits_per_image = torch.matmul(final_image_embeds, final_text_embeds.t()) * logit_scale
        
        return logits_per_image
