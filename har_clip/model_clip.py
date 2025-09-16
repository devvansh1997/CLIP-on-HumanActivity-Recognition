# har_clip/model_clip.py
from dataclasses import dataclass
from typing import Optional, List
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoProcessor

@dataclass
class CLIPCfg:
    name: str = "openai/clip-vit-base-patch32"
    max_txt_len: int = 77
    device: str = "cuda"

class CLIPModel:
    """
    CLIP wrapper for XBM + grad accumulation.
    """
    def __init__(self, cfg: CLIPCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(cfg.name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(cfg.name)

    def train(self): self.model.train()
    def eval(self):  self.model.eval()

    def forward_loss(self, batch):
        # Optional path using HF's built-in CLIP loss
        keep = ("input_ids","attention_mask","pixel_values")
        batch = {k: v.to(self.device) for k, v in batch.items() if k in keep}
        out = self.model(**batch, return_loss=True)
        return out.loss

    # ---- Training-time encoders (no @torch.no_grad) ----
    def image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.model.get_image_features(pixel_values=pixel_values.to(self.device))
        return F.normalize(feats, dim=-1)

    def text_features_from_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        feats = self.model.get_text_features(
            input_ids=input_ids.to(self.device),
            attention_mask=attention_mask.to(self.device)
        )
        return F.normalize(feats, dim=-1)

    # ---- Eval helper from raw strings ----
    @torch.no_grad()
    def text_features(self, texts: List[str]) -> torch.Tensor:
        toks = self.processor(
            text=list(texts),
            padding="max_length",
            max_length=self.cfg.max_txt_len,
            return_tensors="pt"
        ).to(self.device)
        feats = self.model.get_text_features(**toks)
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def logit_scale(self) -> torch.Tensor:
        return self.model.logit_scale.exp()
