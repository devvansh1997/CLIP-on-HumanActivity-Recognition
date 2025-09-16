import os
from typing import List, Dict
import torch
from sklearn.metrics import f1_score, accuracy_score

def _get_class_names(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

@torch.no_grad()
def evaluate_clip(model, test_loader, data_root_test: str, logger=None) -> Dict[str, float]:
    model.eval()
    device = model.device

    class_names = _get_class_names(data_root_test)
    prompts = [f"This is a photo of {c.replace('_',' ')}." for c in class_names]
    T = model.text_features(prompts)  # (C, D)

    y_true, y_pred = [], []
    for batch in test_loader:
        pv = batch["pixel_values"].to(device)
        I  = model.image_features(pv)       # (B, D)
        logits = I @ T.t()                  # (B, C)
        pred = logits.argmax(dim=1)
        y_pred += pred.tolist()
        y_true += batch["label_idx"].tolist()

    f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))
    if logger: logger({"eval/f1_macro": f1, "eval/acc": acc})
    print(f"[Eval] F1(macro): {f1:.4f} | Acc: {acc:.4f}")
    return {"f1_macro": f1, "acc": acc}
