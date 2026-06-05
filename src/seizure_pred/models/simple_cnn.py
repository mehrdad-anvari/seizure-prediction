from __future__ import annotations

import torch
from torch import nn

from seizure_pred.core.config import ModelConfig
from seizure_pred.training.registries import MODELS
from .api import ModelOutput


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 2, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden, 1 if num_classes == 2 else num_classes)

    def forward(self, x: torch.Tensor, *, return_output: bool = False):
        # x: (B,C,T)
        h = self.net(x).squeeze(-1)  # (B,hidden)
        logits = self.head(h)  # (B,1) or (B,K)
        if logits.shape[-1] == 1:
            logits = logits.squeeze(-1)  # (B,)
        if return_output:
            return ModelOutput(logits=logits, aux={"embedding": h})
        return logits


@MODELS.register("simple_cnn", help="Baseline 1D CNN with global average pooling.")
def build_simple_cnn(cfg: ModelConfig) -> nn.Module:
    if cfg.in_channels is None:
        raise ValueError("ModelConfig.in_channels must be set for simple_cnn")
    hidden = int(cfg.kwargs.get("hidden", 64))
    return SimpleCNN(in_channels=int(cfg.in_channels), num_classes=int(cfg.num_classes), hidden=hidden)
