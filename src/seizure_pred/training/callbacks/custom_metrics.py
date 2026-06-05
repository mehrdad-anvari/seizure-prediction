from __future__ import annotations

from typing import Any, Dict

import torch

from seizure_pred.training.engine.callbacks import Callback
from seizure_pred.training.registries import CALLBACKS


class CustomMetrics(Callback):
    """Example callback that derives extra metrics from `state['val_out']`.

    Trainer is expected to set:
      state['val_out'] = {'val_logits': Tensor, 'val_targets': Tensor, ...}
    """

    def __init__(self, name: str = "custom_metrics"):
        self.name = name

    def on_val_end(self, state: Dict[str, Any]) -> None:
        out = state.get("val_out") or {}
        logits = out.get("val_logits")
        targets = out.get("val_targets")
        if logits is None or targets is None:
            return

        logits = logits.detach().cpu().reshape(-1)
        targets = targets.detach().cpu().to(torch.int64).reshape(-1)

        pos = logits[targets == 1]
        neg = logits[targets == 0]

        logs = state.setdefault("logs", {})
        if pos.numel() > 0:
            logs["val_pos_logit_mean"] = float(pos.mean().item())
        if neg.numel() > 0:
            logs["val_neg_logit_mean"] = float(neg.mean().item())


@CALLBACKS.register("custom_metrics", help="Example callback that adds extra validation metrics.")
def build_custom_metrics(**kwargs) -> CustomMetrics:
    return CustomMetrics(**kwargs)
