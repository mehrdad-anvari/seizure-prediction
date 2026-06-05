from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch
from torch import nn

Tensor = torch.Tensor


@runtime_checkable
class SupportsLogits(Protocol):
    def forward(self, x: Tensor) -> Tensor: ...


class BaseModel(nn.Module):
    """Base class for models in this library.

    Convention:
      - forward(x) returns logits
      - binary: logits shape (B,) or (B,1)
      - multiclass: logits shape (B, K)
    """

    def __init__(self):
        super().__init__()
