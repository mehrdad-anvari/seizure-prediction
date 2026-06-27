from __future__ import annotations

from typing import Dict, Optional

import torch

from seizure_pred.training.registries import CALLBACKS


class MyCallback:
    """Template for a new callback.

    Supported hook names (implement any subset):
      - on_fit_start(trainer=..., **ctx)
      - on_fit_end(...)
      - on_epoch_start(epoch=..., **ctx)
      - on_epoch_end(epoch=..., logs=dict, **ctx) -> dict[str,float] optional
      - on_train_batch_end(...)
      - on_val_epoch_start(...)
      - on_val_batch_end(logits, targets, loss, **ctx)
      - on_val_epoch_end(logs=dict, **ctx) -> dict[str,float] optional

    Return value:
      - If a hook returns a dict[str,float], Trainer will merge it into logs/history.
    """

    def __init__(self, **kwargs):
        ...

    def on_val_epoch_end(self, logs: Dict[str, float], **ctx) -> Dict[str, float]:
        # return extra metrics
        return {}


@CALLBACKS.register("my_callback", help="Describe what your callback does")
def build_my_callback(cfg=None, **kwargs):
    return MyCallback(**kwargs)
