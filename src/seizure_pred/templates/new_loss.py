"""Template: add a new loss.

Copy to: seizure_pred/training/components/<your_loss>.py and register into LOSSES.
"""

from __future__ import annotations

from torch import nn

from seizure_pred.core.config import LossConfig
from seizure_pred.training.registries import LOSSES


@LOSSES.register("my_loss", help="Example loss template")
def build_loss(cfg: LossConfig) -> nn.Module:
    # return nn.Module(loss_fn)
    return nn.MSELoss()
