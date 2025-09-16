import argparse, os
import torch
from .dataset import create_dataloader
from .model_siglip import SigLIPCfg, SigLIPModel
from .train import train_siglip
from .evaluate import evaluate_siglip
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
    set_seed(cfg["logging"]["seed"])

    out_dir = cfg["logging"]["out_dir"]
    ensure_dir(out_dir)
    log = default_logger(out_dir)

    # Build model + optim
    mcfg = SigLIPCfg(
        name=cfg["model"]["name"],
        max_txt_len=64,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = SigLIPModel(mcfg)
    optimizer = torch.optim.AdamW(
        model.model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )


    # Dataloaders
    train_loader = create_dataloader(cfg, split="train") if args.mode in ("train","train+eval") else None
    test_loader  = create_dataloader(cfg, split="test")

    # Optionally resume
    if args.ckpt_path:
        print(f"Loading checkpoint: {args.ckpt_path}")
        load_ckpt(model, optimizer if args.mode!="eval" else None, args.ckpt_path)

    # Train
    if args.mode in ("train","train+eval"):
        train_siglip(
            model, train_loader, optimizer,
            epochs=cfg["training"]["epochs"],
            fp16=cfg["training"]["fp16"],
            log_every=cfg["logging"]["print_every"],
            log_fn=log
        )
        ckpt = save_ckpt(model, optimizer, out_dir)
        print(f"Saved checkpoint to {ckpt}")

    # Eval
    if args.mode in ("eval","train+eval"):
        test_root = os.path.join(cfg["data"]["path"], "test")
        evaluate_siglip(model, test_loader, test_root, logger=log)

if __name__ == "__main__":
    main()
