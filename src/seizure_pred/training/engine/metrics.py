from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

Tensor = torch.Tensor


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

    Returns: acc, precision, recall, f1 and confusion counts.
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

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }
