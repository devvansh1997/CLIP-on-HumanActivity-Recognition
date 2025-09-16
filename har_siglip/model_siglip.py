# har_siglip/model_siglip.py
from dataclasses import dataclass
from typing import Optional, List
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor


@dataclass
class SigLIPCfg:
    name: str = "google/siglip-base-patch16-224"
    max_txt_len: int = 64
    device: str = "cuda"


class SigLIPModel:
    """
    Minimal SigLIP wrapper for training with XBM + gradient accumulation.

    Exposes:
      - forward_loss(batch): optional (uses HF built-in pairwise loss)
      - image_features(pixel_values): (B, D) L2-normalized, REQUIRES GRAD (no @torch.no_grad)
      - text_features_from_tokens(input_ids, attention_mask=None): (B, D) L2-normalized, REQUIRES GRAD
      - text_features(texts): (N, D) L2-normalized (decorated with @torch.no_grad for eval convenience)
    """
    def __init__(self, cfg: SigLIPCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(cfg.name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(cfg.name)

    def train(self): 
        self.model.train()

    def eval(self):  
        self.model.eval()

    def forward_loss(self, batch):
        """
        Keeps compatibility with a simple training path that uses SigLIP's built-in loss.
        Not used by the XBM trainer but handy for debugging.
        """
        keep = ("input_ids", "attention_mask", "pixel_values")
        batch = {k: v.to(self.device) for k, v in batch.items() if k in keep}
        out = self.model(**batch, return_loss=True)
        return out.loss

    # ---- Training-time encoders: MUST keep graph (no @torch.no_grad) ----
    def image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: (B, 3, H, W) on any device; moved to self.device inside.
        Returns L2-normalized image embeddings with gradients.
        """
        feats = self.model.get_image_features(pixel_values=pixel_values.to(self.device))
        return F.normalize(feats, dim=-1)

    def text_features_from_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        input_ids: (B, L)
        attention_mask: optional (B, L). If None, uses all-ones mask.
        Returns L2-normalized text embeddings with gradients.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        feats = self.model.get_text_features(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device)
        )
        return F.normalize(feats, dim=-1)

    # ---- Eval-time helper from raw strings (safe to no_grad) ----
    @torch.no_grad()
    def text_features(self, texts: List[str]) -> torch.Tensor:
        """
        texts: list of strings. Uses padding='max_length' to mirror SigLIP pretraining.
        Returns L2-normalized text embeddings (no gradients).
        """
        toks = self.processor(
            text=list(texts),
            padding="max_length",
            max_length=self.cfg.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        feats = self.model.get_text_features(**toks)
        return F.normalize(feats, dim=-1)
