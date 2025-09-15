from dataclasses import dataclass
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
    Minimal SigLIP wrapper:
      - forward_loss(batch) uses SigLIP's built-in pairwise loss (return_loss=True)
      - image_features(pixel_values) and text_features(texts) return L2-normalized embeddings
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
        # batch comes from dataset.py and already includes tokenized text + pixel_values
        # (e.g., input_ids, attention_mask, pixel_values).
        batch = {k: v.to(self.device) for k, v in batch.items() if k in ("input_ids","attention_mask","pixel_values")}
        out = self.model(**batch, return_loss=True)
        return out.loss

    @torch.no_grad()
    def image_features(self, pixel_values: torch.Tensor):
        i = self.model.get_image_features(pixel_values=pixel_values.to(self.device))
        return F.normalize(i, dim=-1)

    @torch.no_grad()
    def text_features(self, texts):
        toks = self.processor(
            text=list(texts),
            padding="max_length",
            max_length=self.cfg.max_txt_len,
            return_tensors="pt",
        ).to(self.device)
        t = self.model.get_text_features(**toks)
        return F.normalize(t, dim=-1)
