from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence, Tuple

import torch


@dataclass
class EvalBatch:
    logits: torch.Tensor
    targets: torch.Tensor
    loss: Optional[torch.Tensor] = None


class Evaluator(Protocol):
    """Evaluator computes metrics from (logits, targets) pairs.

    Contract:
      - update(logits, targets, loss=None) is called for each batch
      - compute() returns a flat dict[str, float]
      - reset() clears internal state

    This is intentionally decoupled from Trainer to allow task-specific
    evaluation (prediction vs detection, MIL vs instance, window->event metrics, etc.).
    """

    def reset(self) -> None: ...
    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: Optional[torch.Tensor] = None) -> None: ...
    def compute(self) -> Dict[str, float]: ...


def _to_cpu_1d(x: torch.Tensor) -> torch.Tensor:
    x = x.detach()
    if x.is_cuda:
        x = x.cpu()
    return x.reshape(-1)
