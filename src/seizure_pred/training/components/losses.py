from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from seizure_pred.training.engine.imbalance import estimate_pos_weight
from seizure_pred.training.registries import LOSSES


@LOSSES.register("bce_logits", help="Binary cross-entropy with logits.")
def build_bce_logits(*, pos_weight: Optional[object] = None, dataset=None, **kwargs) -> nn.Module:
    # pos_weight can be float/tensor or "auto" (requires dataset)
    pw = None
    if pos_weight is not None:
        if isinstance(pos_weight, str) and pos_weight.lower() == "auto":
            if dataset is None:
                raise ValueError("pos_weight='auto' requires dataset=... passed to loss builder")
            pw = estimate_pos_weight(dataset).to(torch.float32)
        else:
            pw = torch.as_tensor(pos_weight, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


@LOSSES.register("focal", help="Focal loss for binary classification (logits).")
def build_focal(*, gamma: float = 2.0, alpha: float = 0.25, reduction: str = "mean", **kwargs) -> nn.Module:
    class _Focal(nn.Module):
        def __init__(self, gamma: float, alpha: float, reduction: str):
            super().__init__()
            self.gamma = float(gamma)
            self.alpha = float(alpha)
            self.reduction = reduction

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            targets = targets.to(logits.dtype)
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            p = torch.sigmoid(logits)
            p_t = p * targets + (1 - p) * (1 - targets)
            loss = bce * ((1 - p_t) ** self.gamma)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
            if self.reduction == "mean":
                return loss.mean()
            if self.reduction == "sum":
                return loss.sum()
            return loss

    return _Focal(gamma=gamma, alpha=alpha, reduction=reduction)


@LOSSES.register("weighted_bce_logits", help="BCEWithLogits with explicit or auto pos_weight.")
def build_weighted_bce_logits(*, pos_weight: object = "auto", dataset=None, **kwargs) -> nn.Module:
    return build_bce_logits(pos_weight=pos_weight, dataset=dataset)
