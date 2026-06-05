from __future__ import annotations

from typing import Optional

import torch

from seizure_pred.training.evaluators.binary import BinaryEvaluator
from seizure_pred.training.registries import EVALUATORS


class MILEvaluator(BinaryEvaluator):
    """MIL evaluator: expects bag logits already aggregated OR accepts (B,bag) logits and aggregates."""

    def __init__(self, aggregation: str = "max", **kwargs):
        super().__init__(**kwargs)
        self.aggregation = aggregation

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: Optional[torch.Tensor] = None) -> None:
        # logits can be shape (B,) or (B, bag)
        if logits.ndim == 2:
            if self.aggregation == "max":
                logits = torch.max(logits, dim=1).values
            elif self.aggregation == "mean":
                logits = torch.mean(logits, dim=1)
            elif self.aggregation == "logsumexp":
                logits = torch.logsumexp(logits, dim=1)
            else:
                raise ValueError(f"Unknown MIL aggregation: {self.aggregation}")
        return super().update(logits=logits, targets=targets, loss=loss)


def register_mil_evaluators() -> None:
    @EVALUATORS.register("mil_binary", help="MIL binary evaluator (aggregates bag logits then computes binary metrics)")
    def _build_mil_evaluator(cfg=None, *, prefix: str = "", **kwargs):
        return MILEvaluator(prefix=prefix, **kwargs)
