from __future__ import annotations

from typing import Any, Dict, Optional

from seizure_pred.training.engine.callbacks import Callback
from seizure_pred.training.registries import CALLBACKS


class LRMonitor(Callback):
    """Logs learning rate into state['logs'] at epoch end."""

    def __init__(self, key: str = "lr"):
        self.key = key

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        opt = state.get("optimizer")
        if opt is None:
            return
        try:
            lr = float(opt.param_groups[0].get("lr"))
        except Exception:
            return
        logs = state.setdefault("logs", {})
        logs[self.key] = lr


@CALLBACKS.register("lr_monitor", help="Log learning rate each epoch.")
def build_lr_monitor(**kwargs) -> LRMonitor:
    return LRMonitor(**kwargs)
