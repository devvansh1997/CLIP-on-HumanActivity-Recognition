#!/usr/bin/env python
import argparse, os, random, time
from typing import List, Tuple
import torch
from PIL import Image
import matplotlib.pyplot as plt

from har_siglip.model_siglip import SigLIPCfg, SigLIPModel
from har_siglip.utils import load_config as load_cfg_sig, load_ckpt as load_ckpt_sig
from har_clip.model_clip import CLIPCfg, CLIPModel
from har_clip.utils import load_config as load_cfg_clip, load_ckpt as load_ckpt_clip
from har_clip.model_clip import CLIPCfg as _CLIPCfg


# ----------------- helpers -----------------
def list_class_names(split_root: str) -> List[str]:
    return sorted([d for d in os.listdir(split_root) if os.path.isdir(os.path.join(split_root, d))])

def pick_random_image(test_root: str) -> Tuple[str, str]:
    classes = list_class_names(test_root)
    if not classes:
        raise FileNotFoundError(f"No class folders under: {test_root}")
    c = random.choice(classes)
    cdir = os.path.join(test_root, c)
    files = [f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if not files:
        raise FileNotFoundError(f"No images in: {cdir}")
    return os.path.join(cdir, random.choice(files)), c

def top1_class(model_text_features, model_image_features, processor, image_path: str, class_names: List[str]) -> str:
    im = Image.open(image_path).convert("RGB")
    prompts = [f"This is a photo of {c.replace('_',' ')}." for c in class_names]
    with torch.no_grad():
        T = model_text_features(prompts)                      # (C, D)
        I = model_image_features(processor(images=im, return_tensors="pt")["pixel_values"])  # (1, D)
        pred_idx = int((I @ T.t()).argmax(dim=1).item())
    return class_names[pred_idx]

def color_for_pred(pred: str, gt: str) -> str:
    return "green" if pred == gt else "red"


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip_config", default="har_siglip/config.yaml")
    ap.add_argument("--clip_config",   default="har_clip/config.yaml")
    ap.add_argument("--siglip_ckpt",   default="", help="Path to SigLIP checkpoint (.pt); defaults to runs/siglip/latest.pt")
    ap.add_argument("--clip_ckpt",     default="", help="Path to CLIP checkpoint (.pt); defaults to runs/clip/latest.pt")
    ap.add_argument("--data_root",     default="", help="Overrides config data.path. Must contain train/ and test/")
    ap.add_argument("--image_path",    default="", help="Optional explicit image path; else pick random from test/")
    ap.add_argument("--save_path",     default="", help="Output PNG; defaults to runs/compare/compare_<ts>.png")
    args = ap.parse_args()

    # Load configs
    cfg_sig = load_cfg_sig(args.siglip_config)
    cfg_clip = load_cfg_clip(args.clip_config)

    # Resolve data root and test split
    data_root = args.data_root or cfg_sig["data"]["path"]
    test_root = os.path.join(data_root, "test")
    if not os.path.isdir(test_root):
        raise FileNotFoundError(f"Test split not found at {test_root}")
    class_names = list_class_names(test_root)

    # Choose image + GT
    if args.image_path:
        image_path = args.image_path
        parts = os.path.normpath(image_path).split(os.sep)
        gt = next((c for c in class_names if c in parts), "unknown")
    else:
        image_path, gt = pick_random_image(test_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ----- SigLIP -----
    sig_model = SigLIPModel(SigLIPCfg(name=cfg_sig["model"]["name"], device=device))
    sig_ckpt = args.siglip_ckpt or os.path.join(cfg_sig["logging"]["out_dir"], "latest.pt")
    if os.path.isfile(sig_ckpt):
        print(f"[SigLIP] loading checkpoint: {sig_ckpt}")
        load_ckpt_sig(sig_model, optimizer=None, ckpt_path=sig_ckpt)
    else:
        print("[SigLIP] no checkpoint found; using base weights.")

    sig_pred = top1_class(sig_model.text_features, sig_model.image_features, sig_model.processor, image_path, class_names)

    # ----- CLIP -----
    from har_clip.model_clip import CLIPCfg as _CLIPCfg
    clip_model = CLIPModel(_CLIPCfg(name=cfg_clip["model"]["name"], device=device))

    clip_ckpt = args.clip_ckpt or os.path.join(cfg_clip["logging"]["out_dir"], "latest.pt")
    if os.path.isfile(clip_ckpt):
        print(f"[CLIP] loading checkpoint: {clip_ckpt}")
        load_ckpt_clip(clip_model, optimizer=None, ckpt_path=clip_ckpt)
    else:
        print("[CLIP] no checkpoint found; using base weights.")

    clip_pred = top1_class(clip_model.text_features, clip_model.image_features, clip_model.processor, image_path, class_names)

    # ----- Render: single image, headers above -----
    im = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.imshow(im); ax.axis("off")

    # Put three lines above the image, centered
    fig.text(0.5, 0.98, f"GT: {gt}", ha="center", va="top", fontsize=14, color="black")
    fig.text(0.5, 0.94, f"SigLIP: {sig_pred}", ha="center", va="top", fontsize=13, color=color_for_pred(sig_pred, gt))
    fig.text(0.5, 0.90, f"CLIP: {clip_pred}",   ha="center", va="top", fontsize=13, color=color_for_pred(clip_pred, gt))

    ts = int(time.time())
    save_path = args.save_path or os.path.join("runs", "compare", f"compare_{ts}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison -> {save_path}")
    print(f"Image: {image_path}\nGT: {gt} | SigLIP: {sig_pred} | CLIP: {clip_pred}")

if __name__ == "__main__":
    main()
