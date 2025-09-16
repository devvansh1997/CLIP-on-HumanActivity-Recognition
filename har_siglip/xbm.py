import torch

class FeatureQueue:
    def __init__(self, dim: int, max_size: int, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_size = int(max_size)
        self.dim = int(dim)
        self.img_mem = torch.empty((0, dim), device=self.device)
        self.txt_mem = torch.empty((0, dim), device=self.device)

    @torch.no_grad()
    def enqueue(self, img_feats: torch.Tensor, txt_feats: torch.Tensor):
        self.img_mem = torch.cat([self.img_mem, img_feats.detach()], dim=0)
        self.txt_mem = torch.cat([self.txt_mem, txt_feats.detach()], dim=0)
        if self.img_mem.size(0) > self.max_size:
            cut = self.img_mem.size(0) - self.max_size
            self.img_mem = self.img_mem[cut:]
            self.txt_mem = self.txt_mem[cut:]

    @torch.no_grad()
    def get(self):
        return self.img_mem, self.txt_mem

    def __len__(self):
        return self.img_mem.size(0)
