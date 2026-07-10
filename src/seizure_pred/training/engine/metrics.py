from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

Tensor = torch.Tensor


def binary_auc(probs: np.ndarray, targets: np.ndarray) -> float:
    """ROC-AUC for binary labels via the rank-based (Mann-Whitney U) statistic.

    Equivalent to the area under the ROC curve and robust to ties because ranks
    are averaged. No sklearn dependency. Returns 0.5 when undefined.
    """
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.int64).ravel()
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Average ranks (1-based); ties share the mean of their ranks.
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1, dtype=np.float64)
    # resolve ties by averaging ranks of equal values
    sorted_p = p[order]
    i = 0
    while i < len(sorted_p):
        j = i
        while j + 1 < len(sorted_p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1

    sum_pos = float(ranks[y == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def binary_average_precision(probs: np.ndarray, targets: np.ndarray) -> float:
    """Average precision (area under PR curve) without sklearn."""
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(targets, dtype=np.int64).ravel()
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return 0.0
    order = np.argsort(-p, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    precision = tp / np.maximum(1.0, tp + fp)
    recall = tp / n_pos
    # stepwise PR area
    ap = float(np.sum((recall[1:] - recall[:-1]) * precision[1:])) if len(recall) > 1 else 0.0
    # include the first point
    ap += float(precision[0] * recall[0])
    return float(min(1.0, max(0.0, ap)))


@dataclass
class MetricState:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    loss_sum: float = 0.0
    n: int = 0

    def update_confusion(self, y_true: Tensor, y_pred: Tensor) -> None:
        y_true = y_true.detach().view(-1).to(torch.int64)
        y_pred = y_pred.detach().view(-1).to(torch.int64)
        self.tp += int(((y_true == 1) & (y_pred == 1)).sum().item())
        self.fp += int(((y_true == 0) & (y_pred == 1)).sum().item())
        self.tn += int(((y_true == 0) & (y_pred == 0)).sum().item())
        self.fn += int(((y_true == 1) & (y_pred == 0)).sum().item())

    def update_loss(self, loss: Tensor, batch_size: int) -> None:
        self.loss_sum += float(loss.detach().item()) * batch_size
        self.n += int(batch_size)

    def compute(self) -> Dict[str, float]:
        eps = 1e-12
        acc = (self.tp + self.tn) / max(1, (self.tp + self.tn + self.fp + self.fn))
        prec = self.tp / max(eps, (self.tp + self.fp))
        rec = self.tp / max(eps, (self.tp + self.fn))
        f1 = (2 * prec * rec) / max(eps, (prec + rec))
        loss = self.loss_sum / max(1, self.n)
        return {"loss": loss, "acc": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}


@torch.no_grad()
def logits_to_pred(logits: Tensor) -> Tensor:
    """Binary by default: threshold sigmoid at 0.5.

    If logits has last dim > 1, uses argmax (multiclass).
    """
    if logits.ndim >= 2 and logits.shape[-1] > 1:
        return torch.argmax(logits, dim=-1)
    probs = torch.sigmoid(logits.view(-1))
    return (probs >= 0.5).to(torch.int64)

@torch.no_grad()
def binary_classification_metrics(logits: Tensor, targets: Tensor, *, threshold: float = 0.5) -> Dict[str, float]:
    """Compute binary classification metrics from logits/targets.

    Returns: acc, precision, recall, f1, auc, ap and confusion counts.

    AUC/AP are computed from sigmoid probabilities (rank-based, no sklearn).
    """
    y_true = targets.detach().view(-1).to(torch.int64)
    probs = torch.sigmoid(logits.detach().view(-1))
    y_pred = (probs >= float(threshold)).to(torch.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum().item())
    fp = int(((y_true == 0) & (y_pred == 1)).sum().item())
    tn = int(((y_true == 0) & (y_pred == 0)).sum().item())
    fn = int(((y_true == 1) & (y_pred == 0)).sum().item())

    eps = 1e-12
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    precision = tp / max(eps, (tp + fp))
    recall = tp / max(eps, (tp + fn))
    f1 = (2 * precision * recall) / max(eps, (precision + recall))

    auc = binary_auc(probs.detach().cpu().numpy(), y_true.detach().cpu().numpy())
    ap = binary_average_precision(probs.detach().cpu().numpy(), y_true.detach().cpu().numpy())

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "ap": float(ap),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
