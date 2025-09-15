from typing import Callable, Optional
import torch
from tqdm import tqdm

def train_siglip(
    model,
    train_loader,
    optimizer: torch.optim.Optimizer,
    epochs: int = 10,
    fp16: bool = True,
    log_every: int = 20,
    log_fn: Optional[Callable[[dict], None]] = None,
):
    scaler = torch.cuda.amp.GradScaler() if (fp16 and torch.cuda.is_available()) else None
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for step, batch in pbar:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model.forward_loss(batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = model.forward_loss(batch)
                loss.backward()
                optimizer.step()

            if (step + 1) % log_every == 0:
                msg = {"train/loss": float(loss.detach().cpu()), "epoch": epoch, "step": step + 1}
                if log_fn: log_fn(msg)
                pbar.set_postfix(loss=msg["train/loss"])
