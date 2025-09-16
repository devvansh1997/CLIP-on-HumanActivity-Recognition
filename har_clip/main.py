import argparse, os
import torch
from .dataset import create_dataloader
from .model_clip import CLIPCfg, CLIPModel
from .train import train_clip
from .evaluate import evaluate_clip
from .utils import load_config, set_seed, default_logger, ensure_dir, save_ckpt, load_ckpt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","eval","train+eval"], default="train")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--ckpt_path", default="")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    trn = cfg["training"]
    # Coerce types in case YAML had quotes
    trn["lr"] = float(trn["lr"]); trn["weight_decay"] = float(trn["weight_decay"])
    trn["mini_batch_size"] = int(trn["mini_batch_size"]); trn["epochs"] = int(trn["epochs"])

    set_seed(cfg["logging"]["seed"])
    out_dir = cfg["logging"]["out_dir"]; ensure_dir(out_dir)
    log = default_logger(out_dir)

    mcfg = CLIPCfg(
        name=cfg["model"]["name"],
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    model = CLIPModel(mcfg)
    optimizer = torch.optim.AdamW(model.model.parameters(),
                                  lr=trn["lr"], weight_decay=trn["weight_decay"])

    train_loader = create_dataloader(cfg, split="train") if args.mode in ("train","train+eval") else None
    test_loader  = create_dataloader(cfg, split="test")

    if args.ckpt_path:
        print(f"Loading checkpoint: {args.ckpt_path}")
        load_ckpt(model, optimizer if args.mode!="eval" else None, args.ckpt_path)

    if args.mode in ("train","train+eval"):
        train_clip(model, train_loader, optimizer,
                   epochs=trn["epochs"],
                   fp16=bool(trn["fp16"]),
                   log_every=cfg["logging"]["print_every"],
                   log_fn=log)
        ckpt = save_ckpt(model, optimizer, out_dir)
        print(f"Saved checkpoint to {ckpt}")

    if args.mode in ("eval","train+eval"):
        test_root = os.path.join(cfg["data"]["path"], "test")
        evaluate_clip(model, test_loader, test_root, logger=log)

if __name__ == "__main__":
    main()
