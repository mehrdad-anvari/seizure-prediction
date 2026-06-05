from __future__ import annotations

from typing import Dict, Optional

import torch

from seizure_pred.training.evaluators.base import _to_cpu_1d
from seizure_pred.training.registries import EVALUATORS


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else 0.0


def _roc_auc_from_scores(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    # y_true, y_score are 1D CPU tensors
    # Sort by descending score
    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order]
    y_score = y_score[order]

    # Count positives/negatives
    P = float(torch.sum(y_true).item())
    N = float(y_true.numel() - torch.sum(y_true).item())
    if P == 0.0 or N == 0.0:
        return 0.0

    # TPR/FPR sweep (vectorized)
    tps = torch.cumsum(y_true, dim=0)
    fps = torch.cumsum(1 - y_true, dim=0)
    tpr = tps / P
    fpr = fps / N

    # Add (0,0) at start
    tpr = torch.cat([torch.tensor([0.0]), tpr])
    fpr = torch.cat([torch.tensor([0.0]), fpr])

    # Trapezoidal integral
    auc = torch.trapz(tpr, fpr).item()
    return float(auc)


class BinaryEvaluator:
    """Standard binary classification evaluator: loss/acc/precision/recall/f1 + optional AUC."""

    def __init__(self, threshold: float = 0.5, compute_auc: bool = True, prefix: str = ""):
        self.threshold = float(threshold)
        self.compute_auc = bool(compute_auc)
        self.prefix = prefix
        self.reset()

    def reset(self) -> None:
        self._loss_sum = 0.0
        self._n = 0
        self._tp = 0
        self._tn = 0
        self._fp = 0
        self._fn = 0
        self._scores = []
        self._targets = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: Optional[torch.Tensor] = None) -> None:
        probs = torch.sigmoid(logits)
        y = _to_cpu_1d(targets).to(torch.int64)
        p = _to_cpu_1d(probs)

        pred = (p >= self.threshold).to(torch.int64)

        tp = int(torch.sum((pred == 1) & (y == 1)).item())
        tn = int(torch.sum((pred == 0) & (y == 0)).item())
        fp = int(torch.sum((pred == 1) & (y == 0)).item())
        fn = int(torch.sum((pred == 0) & (y == 1)).item())

        self._tp += tp
        self._tn += tn
        self._fp += fp
        self._fn += fn
        self._n += int(y.numel())

        if loss is not None:
            self._loss_sum += float(loss.detach().cpu().item()) * int(y.numel())

        if self.compute_auc:
            self._scores.append(p)
            self._targets.append(y)

    def compute(self) -> Dict[str, float]:
        acc = _safe_div(self._tp + self._tn, self._n)
        prec = _safe_div(self._tp, (self._tp + self._fp))
        rec = _safe_div(self._tp, (self._tp + self._fn))
        f1 = _safe_div(2 * prec * rec, (prec + rec))
        out = {
            f"{self.prefix}loss": _safe_div(self._loss_sum, self._n) if self._n else 0.0,
            f"{self.prefix}acc": acc,
            f"{self.prefix}precision": prec,
            f"{self.prefix}recall": rec,
            f"{self.prefix}f1": f1,
        }
        if self.compute_auc and self._targets:
            y_true = torch.cat(self._targets).to(torch.float32)
            y_score = torch.cat(self._scores).to(torch.float32)
            out[f"{self.prefix}auc"] = _roc_auc_from_scores(y_true, y_score)
        return out


def register_binary_evaluators() -> None:
    @EVALUATORS.register("binary", help="Binary classifier evaluator (loss/acc/precision/recall/f1 + optional AUC)")
    def _build_binary_evaluator(cfg=None, *, prefix: str = "", **kwargs):
        # cfg is optional; kwargs may include threshold/compute_auc
        return BinaryEvaluator(prefix=prefix, **kwargs)
