"""Template: add a new model.

Copy to: seizure_pred/models/<your_model>.py and register into MODELS.
"""

from __future__ import annotations

import torch
from torch import nn

from seizure_pred.core.config import ModelConfig
from seizure_pred.training.registries import MODELS


class MyModel(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1, **kwargs):
        super().__init__()
        self.net = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Return raw logits. For binary: shape (B,) or (B,1)
        return self.net(x)


@MODELS.register("my_model", help="Example model template")
def build_my_model(cfg: ModelConfig) -> nn.Module:
    if cfg.in_channels is None:
        raise ValueError("cfg.in_channels must be set")
    return MyModel(in_channels=cfg.in_channels, num_classes=cfg.num_classes, **cfg.kwargs)
