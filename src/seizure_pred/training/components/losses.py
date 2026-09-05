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


class PreictalWeightedLoss(nn.Module):
    """Loss with temporal weighting for preictal samples near seizure onset."""
    def __init__(self, base_loss_fn: nn.Module, max_weight: float = 5.0):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.max_weight = max_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, temporal_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        loss = self.base_loss_fn(logits, targets) # (B,)
        
        if temporal_weights is not None:
            sample_weights = torch.ones_like(targets, dtype=torch.float32)
            preictal_mask = (targets == 1.0)
            
            if preictal_mask.any():
                sample_weights[preictal_mask] = 1.0 + (self.max_weight - 1.0) * temporal_weights[preictal_mask]
            
            loss = loss * sample_weights
            
        return loss.mean()


@LOSSES.register("preictal_weighted", help="BCE or Focal loss with preictal temporal weighting.")
def build_preictal_weighted(*, base_loss: str = "bce_logits", max_weight: float = 5.0, **kwargs) -> nn.Module:
    if base_loss == "bce_logits":
        base_loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    elif base_loss == "focal":
        base_loss_fn = build_focal(reduction="none", **kwargs)
    else:
        raise ValueError(f"Unknown base loss for preictal weighting: {base_loss}")
        
    return PreictalWeightedLoss(base_loss_fn, max_weight=max_weight)


class MILConfidentLoss(nn.Module):
    """MIL Confident Loss supporting multi-class instance logits."""
    needs_class_logits = True

    def __init__(self):
        super().__init__()

    def forward(self, outputs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # outputs: (B, N, 2), y: (B,)
        B, N, C = outputs.shape
        assert C == 2, f"Expected 2 classes, got {C}"
        
        total_bag_loss = 0.0
        for i in range(B):
            bag_label = y[i].long()
            if bag_label == 1:
                bag_mean = torch.mean(outputs[i], dim=0, keepdim=True)  # (1, 2)
                bag_loss = torch.nn.functional.cross_entropy(bag_mean, bag_label.unsqueeze(0)) * 2
            else:
                bag_labels = torch.zeros(N, dtype=torch.long, device=outputs.device)
                bag_loss = torch.nn.functional.cross_entropy(outputs[i], bag_labels)
            total_bag_loss += bag_loss
            
        return total_bag_loss / B


@LOSSES.register("mil_confident_loss", help="MIL Confident Loss on instance-level logits.")
def build_mil_confident_loss(**kwargs) -> nn.Module:
    return MILConfidentLoss()


class CapsuleMarginLoss(nn.Module):
    r"""Margin loss for capsule outputs (Sabour et al.; used by MD-ResCapsNet).

    :math:`L_k = T_k\,\max(0, m^+ - \|v_k\|)^2 + \lambda (1 - T_k)\,\max(0, \|v_k\| - m^-)^2`.

    Capsule models in this library emit a logit transform of the capsule norm so
    they satisfy the logits contract, so the norms are recovered here with a
    sigmoid. That makes this loss a drop-in alternative to ``bce_logits``.
    """

    def __init__(self, m_pos: float = 0.9, m_neg: float = 0.1, lambda_neg: float = 0.5):
        super().__init__()
        self.m_pos = float(m_pos)
        self.m_neg = float(m_neg)
        self.lambda_neg = float(lambda_neg)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        lengths = torch.sigmoid(logits)
        if lengths.ndim == 1:
            lengths = lengths.unsqueeze(-1)

        if lengths.shape[-1] == 1:
            present = targets.reshape(lengths.shape).to(lengths.dtype)
        else:
            present = torch.zeros_like(lengths)
            present.scatter_(1, targets.reshape(-1, 1).long(), 1.0)

        pos = present * torch.nn.functional.relu(self.m_pos - lengths).pow(2)
        neg = self.lambda_neg * (1.0 - present) * torch.nn.functional.relu(lengths - self.m_neg).pow(2)
        return (pos + neg).sum(dim=-1).mean()


@LOSSES.register("capsule_margin", help="Margin loss on capsule norms (MD-ResCapsNet).")
def build_capsule_margin(*, m_pos: float = 0.9, m_neg: float = 0.1,
                         lambda_neg: float = 0.5, **kwargs) -> nn.Module:
    return CapsuleMarginLoss(m_pos=m_pos, m_neg=m_neg, lambda_neg=lambda_neg)
