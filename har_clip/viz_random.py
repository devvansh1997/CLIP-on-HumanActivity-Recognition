# har_clip/viz_random.py
import argparse, os, random
from typing import List
import torch
from PIL import Image
import matplotlib.pyplot as plt

from .model_clip import CLIPCfg, CLIPModel
from .utils import load_config
from .evaluate import _get_class_names  # reuse helper


def _build_text_matrix(model: CLIPModel, class_names: List[str]):
    prompts = [f"This is a photo of {c.replace('_',' ')}." for c in class_names]
    return model.text_features(prompts)


def predict_single(model: CLIPModel, image_path: str, class_names: List[str]):
    im = Image.open(image_path).convert("RGB")
    proc = model.processor(images=im, return_tensors="pt")
    with torch.no_grad():
        I = model.image_features(proc["pixel_values"])  # (1, D)
        T = _build_text_matrix(model, class_names)      # (C, D)
        logits = I @ T.t()                              # (1, C)
        pred_idx = int(logits.argmax(dim=1).item())
        score = float(torch.softmax(logits, dim=1).max().item())
    return pred_idx, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="har_clip/config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_root = os.path.join(cfg["data"]["path"], "test")

    class_names = _get_class_names(data_root)
    c = random.choice(class_names)
    cdir = os.path.join(data_root, c)
    files = [f for f in os.listdir(cdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    image_path = os.path.join(cdir, random.choice(files))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel(CLIPCfg(name=cfg["model"]["name"], device=device))

    pred_idx, score = predict_single(model, image_path, class_names)
    pred_cls = class_names[pred_idx]
    gt_cls = c

    im = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(6, 6))
    plt.imshow(im)
    plt.axis("off")
    plt.title(f"GT: {gt_cls} | Pred: {pred_cls} (p≈{score:.2f})")
    out_dir = cfg["logging"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "viz_random_clip.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization -> {out_path}")


if __name__ == "__main__":
    main()
