# har_clip/train.py
from typing import Optional, Callable
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .xbm import FeatureQueue

def _clip_infonce_with_memory(
    i_feats: torch.Tensor,
    t_feats: torch.Tensor,
    mem_i: Optional[torch.Tensor],
    mem_t: Optional[torch.Tensor],
    logit_scale: torch.Tensor
) -> torch.Tensor:
    """
    CLIP InfoNCE with memory (both directions).
    Current (B) vs concat(current, memory) (B+M).
    """
    B = i_feats.size(0)

    if (mem_i is not None and mem_i.numel() > 0) and (mem_t is not None and mem_t.numel() > 0):
        all_i = torch.cat([i_feats, mem_i], dim=0)  # (B+M, D)
        all_t = torch.cat([t_feats, mem_t], dim=0)  # (B+M, D)
    else:
        all_i, all_t = i_feats, t_feats

    scale = logit_scale
    logits_i2t = (i_feats @ all_t.t()) * scale      # (B, B+M)
    logits_t2i = (t_feats @ all_i.t()) * scale      # (B, B+M)
    targets = torch.arange(B, device=i_feats.device)

    loss_i2t = F.cross_entropy(logits_i2t, targets)
    loss_t2i = F.cross_entropy(logits_t2i, targets)
    return 0.5 * (loss_i2t + loss_t2i)

def train_clip_xbm(
    model,
    train_loader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    fp16: bool,
    accum_steps: int,
    xbm_size: int,
    log_every: int,
    log_fn: Optional[Callable[[dict], None]] = None,
):
    """
    Training loop with gradient accumulation + cross-batch memory for CLIP.
    """
    device = model.device
    scaler = torch.amp.GradScaler('cuda') if (fp16 and torch.cuda.is_available()) else None

    # Init queue dim via a dry pass
    if xbm_size > 0:
        model.eval()
        first = next(iter(train_loader))
        with torch.no_grad():
            pv0 = first["pixel_values"].to(device)
            i0 = model.image_features(pv0)
        xbm = FeatureQueue(dim=i0.size(1), max_size=int(xbm_size), device=str(device))
        model.train()
    else:
        xbm = None
        model.train()

    optimizer.zero_grad(set_to_none=True)
    micro_step = 0

    for epoch in range(1, epochs + 1):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for step, batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids    = batch["input_ids"].to(device)
            attn         = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(device)

            # forward
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    i_feats = model.image_features(pixel_values)
                    t_feats = model.text_features_from_tokens(input_ids, attn)
                    mem_i, mem_t = (xbm.get() if xbm is not None and len(xbm) > 0 else (None, None))
                    loss = _clip_infonce_with_memory(i_feats, t_feats, mem_i, mem_t, model.logit_scale())
                    loss_b = loss / accum_steps
                scaler.scale(loss_b).backward()
            else:
                i_feats = model.image_features(pixel_values)
                t_feats = model.text_features_from_tokens(input_ids, attn)
                mem_i, mem_t = (xbm.get() if xbm is not None and len(xbm) > 0 else (None, None))
                loss = _clip_infonce_with_memory(i_feats, t_feats, mem_i, mem_t, model.logit_scale())
                (loss / accum_steps).backward()

            micro_step += 1
            if micro_step % accum_steps == 0:
                if scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if xbm is not None:
                    with torch.no_grad():
                        xbm.enqueue(i_feats.detach(), t_feats.detach())

            if (step + 1) % log_every == 0:
                val = float(loss.detach().cpu())
                if log_fn: log_fn({"train/loss": val, "epoch": epoch, "step": step + 1})
                pbar.set_postfix(loss=val)
