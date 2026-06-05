from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class ModelOutput:
    """Standardized output from seizure_pred models.

    logits: raw, unnormalized scores.
      - For binary classification: shape (B,) or (B,1)
      - For multiclass: shape (B,K)

    aux: optional dict for extra outputs (attention maps, embeddings, etc.).
    """
    logits: torch.Tensor
    aux: Dict[str, Any] = None

    def __post_init__(self):
        if self.aux is None:
            self.aux = {}

    def probs(self) -> torch.Tensor:
        """Convert logits to probabilities (best effort)."""
        if self.logits.ndim == 1 or (self.logits.ndim == 2 and self.logits.shape[-1] == 1):
            return torch.sigmoid(self.logits)
        return torch.softmax(self.logits, dim=-1)


class ModelProtocol(torch.nn.Module):
    """Optional protocol-like base for models.

    You don't have to inherit from this, but following the contract helps:
      forward(x) -> torch.Tensor | ModelOutput
    """
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor | ModelOutput:  # pragma: no cover
        raise NotImplementedError
