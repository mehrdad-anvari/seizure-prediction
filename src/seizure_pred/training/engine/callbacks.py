from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


class Callback:
    """Base class for training callbacks.

    All methods are optional. Override what you need.
    The `state` dict is mutable and shared across callbacks.
    """

    # Train lifecycle
    def on_train_start(self, state: Dict[str, Any]) -> None:
        pass

    def on_train_end(self, state: Dict[str, Any]) -> None:
        pass

    # Epoch lifecycle
    def on_epoch_start(self, state: Dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        pass

    # Batch lifecycle (train)
    def on_batch_end(self, state: Dict[str, Any]) -> None:
        pass

    # Validation lifecycle
    def on_val_start(self, state: Dict[str, Any]) -> None:
        pass

    def on_val_batch_end(self, state: Dict[str, Any]) -> None:
        pass

    def on_val_end(self, state: Dict[str, Any]) -> None:
        pass


class CallbackList:
    """Thin dispatcher over a list of callbacks."""

    def __init__(self, callbacks: Optional[Iterable[Callback]] = None):
        self.callbacks: List[Callback] = list(callbacks) if callbacks is not None else []

    def on_train_start(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_start(state)

    def on_train_end(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_end(state)

    def on_epoch_start(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(state)

    def on_epoch_end(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(state)

    def on_batch_end(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(state)

    def on_val_start(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_val_start(state)

    def on_val_batch_end(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_val_batch_end(state)

    def on_val_end(self, state: Dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_val_end(state)
