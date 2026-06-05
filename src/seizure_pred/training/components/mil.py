from __future__ import annotations

import torch
from torch import nn

from seizure_pred.training.registries import LOSSES


class BagAggregator(nn.Module):
    """Aggregate instance logits in a bag to a bag logit.

    Supported:
      - max: max over instances
      - mean: mean over instances
      - logsumexp: smooth max
    """

    def __init__(self, mode: str = "max"):
        super().__init__()
        mode = mode.lower()
        if mode not in {"max", "mean", "logsumexp"}:
            raise ValueError(f"Unknown aggregation mode: {mode}")
        self.mode = mode

    def forward(self, instance_logits: torch.Tensor) -> torch.Tensor:
        # instance_logits: (B, bag, ...)
        if self.mode == "max":
            return instance_logits.max(dim=1).values
        if self.mode == "mean":
            return instance_logits.mean(dim=1)
        return torch.logsumexp(instance_logits, dim=1)


@LOSSES.register("mil_bce_logits", help="MIL BCEWithLogitsLoss on aggregated bag logits")
class MILBCEWithLogitsLoss(nn.Module):
    def __init__(self, aggregation: str = "max", pos_weight: float | None = None):
        super().__init__()
        self.agg = BagAggregator(aggregation)
        pw = None
        if pos_weight is not None:
            pw = torch.tensor([float(pos_weight)])
        self.crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, instance_logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # instance_logits: (B, bag) or (B, bag, 1)
        if instance_logits.dim() == 3 and instance_logits.size(-1) == 1:
            instance_logits = instance_logits.squeeze(-1)
        bag_logits = self.agg(instance_logits)
        if bag_logits.dim() > 1:
            bag_logits = bag_logits.squeeze(-1)
        y = y.float()
        return self.crit(bag_logits, y)
