# har_siglip/train.py
from typing import Optional, Callable
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .xbm import FeatureQueue


def _bce_pairwise_loss_with_memory(
    i_feats: torch.Tensor,
    t_feats: torch.Tensor,
    mem_t_feats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    SigLIP-style pairwise sigmoid loss (image -> text):
      - logits = [i @ t_cur^T | i @ t_mem^T]  -> (B, B+M)
      - targets: 1.0 on diagonal of first B columns; 0 elsewhere.
      - Use BCEWithLogits with a positive weight to balance many negatives.
    """
    B = i_feats.size(0)

    if mem_t_feats is not None and mem_t_feats.numel() > 0:
        all_t = torch.cat([t_feats, mem_t_feats], dim=0)  # (B+M, D)
    else:
        all_t = t_feats  # (B, D)

    logits_i2t = i_feats @ all_t.t()                      # (B, B+M)

    target = torch.zeros_like(logits_i2t)
    # positives are the diagonal in the first B columns
    target[:, :B].fill_diagonal_(1.0)

    pos_weight = torch.tensor(max(logits_i2t.size(1) - 1.0, 1.0), device=logits_i2t.device)
    loss = F.binary_cross_entropy_with_logits(logits_i2t, target, pos_weight=pos_weight)
    return loss


def train_siglip_xbm(
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
    Training loop with:
      - gradient accumulation (accum_steps)
      - cross-batch memory (XBM) holding text/image features

    Notes:
      * model.image_features(...) and model.text_features_from_tokens(...) MUST build a graph
        (i.e., do NOT decorate them with @torch.no_grad()).
      * We only detach when enqueuing into the memory queue.
    """
    device = model.device
    scaler = torch.amp.GradScaler('cuda') if (fp16 and torch.cuda.is_available()) else None

    # Initialize memory queue (if requested) and infer feature dim with a single dry pass.
    if xbm_size > 0:
        model.eval()
        first_batch = next(iter(train_loader))
        with torch.no_grad():
            pv0 = first_batch["pixel_values"].to(device)
            i0 = model.image_features(pv0)  # no grad here; this is just to get D
        xbm = FeatureQueue(dim=i0.size(1), max_size=int(xbm_size), device=str(device))
        model.train()
    else:
        xbm = None
        model.train()

    optimizer.zero_grad(set_to_none=True)
    micro_step = 0  # counts micro-batches toward one optimizer step

    for epoch in range(1, epochs + 1):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
        for step, batch in pbar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids    = batch["input_ids"].to(device)
            attn         = batch.get("attention_mask", None)
            if attn is not None:
                attn = attn.to(device)

            # ---- forward (with/without autocast) ----
            if scaler:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    i_feats = model.image_features(pixel_values)  # (B, D), requires grad
                    t_feats = model.text_features_from_tokens(input_ids, attn)  # (B, D), requires grad
                    mem_i, mem_t = (xbm.get() if xbm is not None and len(xbm) > 0 else (None, None))
                    loss = _bce_pairwise_loss_with_memory(i_feats, t_feats, mem_t)
                    loss_for_backward = loss / accum_steps
                scaler.scale(loss_for_backward).backward()
            else:
                i_feats = model.image_features(pixel_values)
                t_feats = model.text_features_from_tokens(input_ids, attn)
                mem_i, mem_t = (xbm.get() if xbm is not None and len(xbm) > 0 else (None, None))
                loss = _bce_pairwise_loss_with_memory(i_feats, t_feats, mem_t)
                loss_for_backward = loss / accum_steps
                loss_for_backward.backward()

            micro_step += 1

            # ---- optimizer step on accumulation boundary ----
            if micro_step % accum_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Update memory AFTER the optimizer step
                if xbm is not None:
                    with torch.no_grad():
                        xbm.enqueue(i_feats.detach(), t_feats.detach())

            # ---- logging ----
            if (step + 1) % log_every == 0:
                loss_item = float(loss.detach().cpu())  # report per-micro (pre-division) loss
                if log_fn:
                    log_fn({"train/loss": loss_item, "epoch": epoch, "step": step + 1})
                pbar.set_postfix(loss=loss_item)
