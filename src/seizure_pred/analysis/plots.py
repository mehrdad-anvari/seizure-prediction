from __future__ import annotations

import json
from typing import Optional

import numpy as np


def _mpl():
    # Lazy import: allows analysis to run without matplotlib installed unless plotting is used.
    # Force a non-interactive backend for headless/CI environments.
    import matplotlib  # type: ignore
    try:
        matplotlib.use("Agg", force=True)  # type: ignore
    except Exception:
        pass
    import matplotlib.pyplot as plt  # type: ignore
    return plt


def plot_history(history_jsonl: str, *, save_path: str) -> None:
    plt = _mpl()

    epochs = []
    train_loss = []
    val_loss = []

    with open(history_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            epochs.append(int(row.get("epoch", len(epochs) + 1)))
            if "train_loss" in row:
                train_loss.append(float(row["train_loss"]))
            if "val_loss" in row:
                val_loss.append(float(row["val_loss"]))

    plt.figure()
    if train_loss:
        plt.plot(epochs[: len(train_loss)], train_loss, label="train_loss")
    if val_loss:
        plt.plot(epochs[: len(val_loss)], val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion(confusion, *, save_path: str) -> None:
    plt = _mpl()
    c = np.asarray(confusion, dtype=float)

    plt.figure()
    plt.imshow(c, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])

    for (i, j), v in np.ndenumerate(c):
        plt.text(j, i, f"{int(v)}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, *, save_path: str, auc: Optional[float] = None) -> None:
    plt = _mpl()
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}" if auc is not None else "ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", label="chance")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pr(rec: np.ndarray, prec: np.ndarray, *, save_path: str, auc: Optional[float] = None) -> None:
    plt = _mpl()
    plt.figure()
    plt.plot(rec, prec, label=f"AUC={auc:.4f}" if auc is not None else "PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
