import os
from typing import List, Dict
import torch
from sklearn.metrics import f1_score, accuracy_score

def _get_class_names_from_split_root(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

@torch.no_grad()
def evaluate_siglip(model, test_loader, data_root_test: str, logger=None) -> Dict[str, float]:
    model.eval()
    device = model.device

    # Build class prompts (once)
    class_names = _get_class_names_from_split_root(data_root_test)
    prompts = [f"This is a photo of {c.replace('_',' ')}." for c in class_names]
    T = model.text_features(prompts)  # (C, D)

    # Predict
    y_true, y_pred = [], []
    for batch in test_loader:
        pv = batch["pixel_values"].to(device)    # (B, 3, H, W), already processed
        I = model.image_features(pv)             # (B, D)
        logits = I @ T.t()                       # (B, C)
        pred = logits.argmax(dim=1)
        y_pred += pred.tolist()
        y_true += batch["label_idx"].tolist()

    f1 = f1_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)
    if logger:
        logger({"eval/f1_macro": f1, "eval/acc": acc})
    print(f"[Eval] F1(macro): {f1:.4f} | Acc: {acc:.4f}")
    return {"f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "acc": float(accuracy_score(y_true, y_pred))}
