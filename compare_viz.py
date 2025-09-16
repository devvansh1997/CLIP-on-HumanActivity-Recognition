#!/usr/bin/env python
import argparse, os, random, time
from typing import List, Tuple
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Local imports from both trees ---
from har_siglip.model_siglip import SigLIPCfg, SigLIPModel
from har_siglip.utils import load_config as load_cfg_sig, load_ckpt as load_ckpt_sig
from har_clip.model_clip import CLIPCfg, CLIPModel
from har_clip.utils import load_config as load_cfg_clip, load_ckpt as load_ckpt_clip

# ---------- helpers ----------
def list_class_names(split_root: str) -> List[str]:
    return sorted([d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d))])

def pick_random_image(test_root: str) -> Tuple[str, str]:
    """returns (image_path, class_name)"""
    classes = list_class_names(test_root)
    c = random.choice(classes)
    cdir = os.path.join(test_root, c)
    files = [f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        raise FileNotFoundError(f"No images in {cdir}")
    return os.path.join(cdir, random.choice(files)), c

def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def topk_from_logits(logits_1xC: torch.Tensor, class_names: List[str], k: int = 3):
    probs = torch.softmax(logits_1xC, dim=1).squeeze(0).cpu().numpy()
    idxs = probs.argsort()[::-1][:k]
    return [(class_names[i], float(probs[i])) for i in idxs]

# ---------- core ----------
@torch.no_grad()
def predict_siglip(model: SigLIPModel, image_path: str, class_names: List[str]):
    im = Image.open(image_path).convert("RGB")
    # text matrix (compute once per call; small C)
    prompts = [f"This is a photo of {c.replace('_',' ')}." for c in class_names]
    T = model.text_features(prompts)                # (C, D)
    proc = model.processor(images=im, return_tensors="pt")
    I = model.image_features(proc["pixel_values"])  # (1, D)
    logits = I @ T.t()                              # (1, C)
    top3 = topk_from_logits(logits, class_names, k=3)
    pred_cls, pred_p = top3[0]
    return pred_cls, pred_p, top3

@torch.no_grad()
def predict_clip(model: CLIPModel, image_path: str, class_names: List[str]):
    im = Image.open(image_path).convert("RGB")
    prompts = [f"This is a photo of {c.replace('_',' ')}." for c in class_names]
    T = model.text_features(prompts)                # (C, D)
    proc = model.processor(images=im, return_tensors="pt")
    I = model.image_features(proc["pixel_values"])  # (1, D)
    logits = I @ T.t()                              # (1, C)
    top3 = topk_from_logits(logits, class_names, k=3)
    pred_cls, pred_p = top3[0]
    return pred_cls, pred_p, top3

def render_comparison(image_path: str, gt: str,
                      sig_pred, clip_pred,
                      save_path: str):
    """
    sig_pred/clip_pred = (pred_cls, pred_p, top3 list)
    """
    im = Image.open(image_path).convert("RGB")

    plt.figure(figsize=(12, 7))
    # left: image
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(im); ax1.axis("off")
    ax1.set_title(f"Image\nGT: {gt}", fontsize=12)

    # right: table of predictions
    ax2 = plt.subplot(1, 2, 2)
    ax2.axis("off")

    (sig_top1, sig_p, sig_top3) = sig_pred
    (clip_top1, clip_p, clip_top3) = clip_pred

    lines = []
    lines.append(("SigLIP Top-1", f"{sig_top1} ({sig_p:.2f})"))
    for i, (c, p) in enumerate(sig_top3, 1):
        lines.append((f"SigLIP Top-{i}", f"{c} ({p:.2f})"))
    lines.append(("", ""))  # spacer
    lines.append(("CLIP Top-1", f"{clip_top1} ({clip_p:.2f})"))
    for i, (c, p) in enumerate(clip_top3, 1):
        lines.append((f"CLIP Top-{i}", f"{c} ({p:.2f})"))

    # render text
    txt = "\n".join([f"{k:>14}: {v}" for (k, v) in lines])
    ax2.text(0.02, 0.98, txt, va="top", family="monospace", fontsize=11)

    plt.suptitle("SigLIP vs CLIP — same test image", fontsize=14)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison -> {save_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip_config", default="har_siglip/config.yaml")
    ap.add_argument("--clip_config",   default="har_clip/config.yaml")
    ap.add_argument("--siglip_ckpt",   default="", help="Path to SigLIP checkpoint (.pt). Defaults to runs/siglip/latest.pt")
    ap.add_argument("--clip_ckpt",     default="", help="Path to CLIP checkpoint (.pt). Defaults to runs/clip/latest.pt")
    ap.add_argument("--image_path",    default="", help="Optional image path; if empty, pick random from test split")
    ap.add_argument("--save_path",     default="", help="Output PNG path; defaults to runs/compare/compare_<ts>.png")
    args = ap.parse_args()

    # Load configs
    cfg_sig = load_cfg_sig(args.siglip_config)
    cfg_clip = load_cfg_clip(args.clip_config)

    # Resolve test root (must be same taxonomy for fair compare)
    test_root = os.path.join(cfg_sig["data"]["path"], "test")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Test split not found at {test_root}")
    class_names = list_class_names(test_root)

    # Pick image + ground truth
    if args.image_path:
        image_path = args.image_path
        # infer gt from path if it’s under class folder; else mark unknown
        parts = os.path.normpath(image_path).split(os.sep)
        gt = next((c for c in class_names if c in parts), "unknown")
    else:
        image_path, gt = pick_random_image(test_root)

    # Build models (optionally load fine-tuned checkpoints)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sig_model = SigLIPModel(SigLIPCfg(name=cfg_sig["model"]["name"], device=device))
    sig_ckpt = args.siglip_ckpt or os.path.join(cfg_sig["logging"]["out_dir"], "latest.pt")
    if os.path.isfile(sig_ckpt):
        print(f"[SigLIP] loading checkpoint: {sig_ckpt}")
        load_ckpt_sig(sig_model, optimizer=None, ckpt_path=sig_ckpt)
    else:
        print("[SigLIP] no checkpoint found; using base weights.")

    clip_model = CLIPModel(CLIPCfg(name=cfg_clip["model"]["name"], device=device))
    clip_ckpt = args.clip_ckpt or os.path.join(cfg_clip["logging"]["out_dir"], "latest.pt")
    if os.path.isfile(clip_ckpt):
        print(f"[CLIP] loading checkpoint: {clip_ckpt}")
        load_ckpt_clip(clip_model, optimizer=None, ckpt_path=clip_ckpt)
    else:
        print("[CLIP] no checkpoint found; using base weights.")

    # Run both on the same image
    sig_pred = predict_siglip(sig_model, image_path, class_names)
    clip_pred = predict_clip(clip_model, image_path, class_names)

    # Save
    ts = int(time.time())
    save_path = args.save_path or os.path.join("runs", "compare", f"compare_{ts}.png")
    render_comparison(image_path, gt, sig_pred, clip_pred, save_path)

if __name__ == "__main__":
    main()
