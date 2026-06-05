from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple, TypedDict

import torch

Tensor = torch.Tensor


class Meta(TypedDict, total=False):
    # free-form metadata: segment index, file path, start/end times, etc.
    subject_id: str
    file: str
    start: float
    end: float
    label: str
    extra: Dict[str, Any]


@dataclass
class Batch:
    """Standard instance batch contract (prediction/detection)."""

    x: Tensor              # (B, C, T) or (B, C, F, T) depending on preprocessing/model
    y: Tensor              # (B,) or (B,1) binary/multi
    meta: List[Dict[str, Any]]


@dataclass
class MilBatch:
    """Multiple-instance learning (MIL) batch contract."""

    x: Tensor              # (B, bag, C, T) (or higher-dim)
    y: Tensor              # (B,)
    meta: List[List[Dict[str, Any]]]
