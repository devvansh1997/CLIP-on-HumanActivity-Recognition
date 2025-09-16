import argparse, os
import torch
from .dataset import create_dataloader
from .model_siglip import SigLIPCfg, SigLIPModel
from .train import train_siglip_xbm
from .evaluate import evaluate_siglip
from .utils import load_config, set_seed, default_logger, ensure_dir, save_ckpt, load_ckpt

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train","eval","train+eval"], default="train")
    ap.add_argument("--config", default="har_siglip/config.yaml")
    ap.add_argument("--ckpt_path", default="")
    return ap.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    tr = cfg["training"]
    # coerce
    tr["lr"] = float(tr["lr"]); tr["weight_decay"] = float(tr["weight_decay"])
    tr["epochs"] = int(tr["epochs"])
    tr["fp16"] = bool(tr["fp16"])
    # support both keys
    mb = tr.get("micro_batch_size", tr.get("mini_batch_size", 32))
    tr["micro_batch_size"] = int(mb)
    tr["accum_steps"] = int(tr.get("accum_steps", 1))
    tr["xbm_size"] = int(tr.get("xbm_size", 0))

    set_seed(cfg["logging"]["seed"])
    out_dir = cfg["logging"]["out_dir"]; ensure_dir(out_dir)
    log = default_logger(out_dir)

    mcfg = SigLIPCfg(
        name=cfg["model"]["name"],
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    model = SigLIPModel(mcfg)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])

    # Build loaders (dataset reads batch size from config["training"]["mini_batch_size"])
    # We'll patch that key so your dataset uses the micro batch size.
    cfg["training"]["mini_batch_size"] = tr["micro_batch_size"]
    train_loader = create_dataloader(cfg, split="train") if args.mode in ("train","train+eval") else None
    test_loader  = create_dataloader(cfg, split="test")

    if args.ckpt_path:
        print(f"Loading checkpoint: {args.ckpt_path}")
        load_ckpt(model, optimizer if args.mode!="eval" else None, args.ckpt_path)

    if args.mode in ("train","train+eval"):
        train_siglip_xbm(
            model, train_loader, optimizer,
            epochs=tr["epochs"], fp16=tr["fp16"],
            accum_steps=tr["accum_steps"], xbm_size=tr["xbm_size"],
            log_every=cfg["logging"]["print_every"], log_fn=log
        )
        ckpt = save_ckpt(model, optimizer, out_dir)
        print(f"Saved checkpoint to {ckpt}")

    if args.mode in ("eval","train+eval"):
        test_root = os.path.join(cfg["data"]["path"], "test")
        evaluate_siglip(model, test_loader, test_root, logger=log)

if __name__ == "__main__":
    main()
