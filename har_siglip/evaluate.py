import os
from typing import List, Dict, Tuple
import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
import matplotlib.pyplot as plt


def _get_class_names_from_split_root(root: str) -> List[str]:
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
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

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
def evaluate_siglip(
    model,
    test_loader,
    data_root_test: str,
    out_dir: str = "runs/siglip",
    logger=None,
) -> Dict[str, float]:
    """
    Runs zero-shot style evaluation:
      - computes class-wise metrics (precision/recall/F1/support)
      - saves confusion matrix PNG
      - saves per-class metrics CSV
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    device = model.device

    # 1) Build class prompts once
    class_names = _get_class_names_from_split_root(data_root_test)
    prompts = [f"This is a photo of {c.replace('_',' ')}." for c in class_names]
    T = model.text_features(prompts)  # (C, D)

    # 2) Predict
    y_true, y_pred = [], []
    for batch in test_loader:
        pv = batch["pixel_values"].to(device)  # (B, 3, H, W)
        I = model.image_features(pv)           # (B, D)
        logits = I @ T.t()                     # (B, C)
        pred = logits.argmax(dim=1)
        y_pred += pred.tolist()
        y_true += batch["label_idx"].tolist()

    # 3) Aggregate metrics
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))
    if logger:
        logger({"eval/f1_macro": f1, "eval/acc": acc})

    # 4) Class-wise report -> CSV
    report = classification_report(
        y_true, y_pred, output_dict=True, target_names=class_names, zero_division=0
    )
    # save CSV
    import csv
    csv_path = os.path.join(out_dir, "per_class_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["class", "precision", "recall", "f1", "support"]
        writer.writerow(header)
        for cname in class_names:
            row = report.get(cname, {})
            writer.writerow([
                cname,
                row.get("precision", 0.0),
                row.get("recall", 0.0),
                row.get("f1-score", 0.0),
                int(row.get("support", 0)),
            ])
        # overall
        writer.writerow([])
        writer.writerow(["macro avg", report["macro avg"]["precision"],
                         report["macro avg"]["recall"], report["macro avg"]["f1-score"],
                         int(sum(report[c]["support"] for c in class_names))])
        writer.writerow(["accuracy", "", "", acc,
                         int(sum(report[c]["support"] for c in class_names))])

    # 5) Confusion matrix -> PNG
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    _plot_confusion_matrix(cm, class_names, cm_path, normalize=True,
                           title="Confusion Matrix (normalized)")

    print(f"[Eval] F1(macro): {f1:.4f} | Acc: {acc:.4f}")
    print(f"Saved per-class CSV -> {csv_path}")
    print(f"Saved confusion matrix -> {cm_path}")
    return {"f1_macro": f1, "acc": acc}
