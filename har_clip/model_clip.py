from dataclasses import dataclass
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
    Thin wrapper around HF CLIP:
      - forward_loss(batch) uses CLIP's built-in contrastive loss (return_loss=True)
      - image_features / text_features return L2-normalized embeddings
    """
    def __init__(self, cfg: CLIPCfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(cfg.name).to(self.device)
        self.processor = AutoProcessor.from_pretrained(cfg.name)

    def train(self): self.model.train()
    def eval(self):  self.model.eval()

    def forward_loss(self, batch):
        keep = ("input_ids", "attention_mask", "pixel_values")
        batch = {k: v.to(self.device) for k, v in batch.items() if k in keep}
        out = self.model(**batch, return_loss=True)  # CLIP InfoNCE loss across batch
        return out.loss

    @torch.no_grad()
    def image_features(self, pixel_values: torch.Tensor):
        feats = self.model.get_image_features(pixel_values=pixel_values.to(self.device))
        return F.normalize(feats, dim=-1)

    @torch.no_grad()
    def text_features(self, texts):
        toks = self.processor(
            text=list(texts),
            padding="max_length",
            max_length=self.cfg.max_txt_len,
            return_tensors="pt"
        ).to(self.device)
        feats = self.model.get_text_features(**toks)
        return F.normalize(feats, dim=-1)
