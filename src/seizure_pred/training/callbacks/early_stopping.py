from __future__ import annotations

from typing import Any, Dict, List, Optional

from seizure_pred.training.engine.callbacks import Callback
from seizure_pred.training.registries import CALLBACKS


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Sets state['stop_requested']=True when triggered.

    ``monitor`` accepts either spelling of a validation metric: the bare name the
    trainer puts in ``state['val_metrics']`` (``auc``, ``f1``, ``loss``) or the
    ``val_``-prefixed name that ``history.jsonl`` records (``val_auc``,
    ``val_loss``).
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

    def _monitored_value(self, state: Dict[str, Any]) -> Optional[float]:
        """Resolve ``self.monitor`` against the trainer state.

        The trainer exposes validation metrics under ``state['val_metrics']``
        with bare names and the losses at the top level of ``state``; nothing
        populates ``state['logs']`` unless another callback does, so that is
        checked first (for backwards compatibility) and then fallen through.
        """
        names: List[str] = [self.monitor]
        if self.monitor.startswith("val_"):
            names.append(self.monitor[len("val_"):])
        else:
            names.append(f"val_{self.monitor}")

        sources: List[Dict[str, Any]] = [
            state.get("logs") or {},
            state.get("val_metrics") or {},
            state,
        ]
        for source in sources:
            for name in names:
                value = source.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                return float(value)
        return None

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        value = self._monitored_value(state)
        if value is None:
            return

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
