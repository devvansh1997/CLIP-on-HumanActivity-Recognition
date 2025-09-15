import os, json, random
import torch
import yaml
from datetime import datetime

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # faster training for fixed input sizes

def load_config(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def default_logger(log_dir: str):
    ensure_dir(log_dir)
    log_path = os.path.join(log_dir, "log.jsonl")
    def _log(d: dict):
        d = {"ts": datetime.utcnow().isoformat(), **d}
        with open(log_path, "a") as f:
            f.write(json.dumps(d) + "\n")
    return _log

def save_ckpt(model, optimizer, out_dir: str, fname: str = "latest.pt"):
    ensure_dir(out_dir)
    path = os.path.join(out_dir, fname)
    torch.save({
        "model": model.model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    return path

def load_ckpt(model, optimizer, ckpt_path: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    model.model.load_state_dict(ck["model"], strict=True)
    if optimizer and "optimizer" in ck:
        optimizer.load_state_dict(ck["optimizer"])
