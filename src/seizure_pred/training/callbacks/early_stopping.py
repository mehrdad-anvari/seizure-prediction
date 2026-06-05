from __future__ import annotations

from typing import Any, Dict

from seizure_pred.training.engine.callbacks import Callback
from seizure_pred.training.registries import CALLBACKS


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Sets state['stop_requested']=True when triggered.
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.0,
    ):
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.best = None
        self.bad_epochs = 0

    def _is_improvement(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "min":
            return value < (self.best - self.min_delta)
        return value > (self.best + self.min_delta)

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        logs = state.get("logs") or {}
        if self.monitor not in logs:
            return
        value = float(logs[self.monitor])

        if self._is_improvement(value):
            self.best = value
            self.bad_epochs = 0
            return

        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            state["stop_requested"] = True
            state["stop"] = True
            state.setdefault("logs", {})["early_stopping_triggered"] = True


@CALLBACKS.register("early_stopping", help="Stop training early based on a monitored metric.")
def build_early_stopping(**kwargs) -> EarlyStopping:
    return EarlyStopping(**kwargs)
