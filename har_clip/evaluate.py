# har_clip/evaluate.py
import os
from typing import List, Dict
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt


def _get_class_names(root: str) -> List[str]:
    return sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])


def _plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    normalize: bool = True,
    title: str = "Confusion Matrix",
):
    if normalize:
        with np.errstate(all="ignore"):
            cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0 if cm.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = format(cm[i, j], fmt)
            plt.text(
                j, i, val,
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


@torch.no_grad()
def evaluate_clip(
    model,
    test_loader,
    data_root_test: str,
    out_dir: str = "runs/clip",
    logger=None,
) -> Dict[str, float]:
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    device = model.device

    class_names = _get_class_names(data_root_test)
    prompts = [f"This is a photo of {c.replace('_',' ')}." for c in class_names]
    T = model.text_features(prompts)  # (C, D)

    y_true, y_pred = [], []
    for batch in test_loader:
        pv = batch["pixel_values"].to(device)
        I = model.image_features(pv)
        logits = I @ T.t()
        pred = logits.argmax(dim=1)
        y_pred += pred.tolist()
        y_true += batch["label_idx"].tolist()

    f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))
    if logger:
        logger({"eval/f1_macro": f1, "eval/acc": acc})

    # CSV report
    report = classification_report(
        y_true, y_pred, output_dict=True, target_names=class_names, zero_division=0
    )
    import csv
    csv_path = os.path.join(out_dir, "per_class_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1", "support"])
        for cname in class_names:
            r = report.get(cname, {})
            w.writerow([
                cname,
                r.get("precision", 0.0),
                r.get("recall", 0.0),
                r.get("f1-score", 0.0),
                int(r.get("support", 0)),
            ])
        w.writerow([])
        w.writerow(["macro avg", report["macro avg"]["precision"],
                    report["macro avg"]["recall"], report["macro avg"]["f1-score"],
                    int(sum(report[c]["support"] for c in class_names))])
        w.writerow(["accuracy", "", "", acc,
                    int(sum(report[c]["support"] for c in class_names))])

    # CM plot
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    _plot_confusion_matrix(cm, class_names, cm_path, normalize=True,
                           title="Confusion Matrix (normalized)")

    print(f"[Eval] F1(macro): {f1:.4f} | Acc: {acc:.4f}")
    print(f"Saved per-class CSV -> {csv_path}")
    print(f"Saved confusion matrix -> {cm_path}")
    return {"f1_macro": f1, "acc": acc}
