from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn


@dataclass
class Checkpoint:
    epoch: int
    step: int
    model_state: Dict[str, Any]
    optim_state: Optional[Dict[str, Any]] = None
    sched_state: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None


def save_checkpoint(path: str | Path, ckpt: Checkpoint) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": ckpt.epoch,
            "step": ckpt.step,
            "model_state": ckpt.model_state,
            "optim_state": ckpt.optim_state,
            "sched_state": ckpt.sched_state,
            "extra": ckpt.extra or {},
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def restore_checkpoint(
    checkpoint_path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    payload = load_checkpoint(checkpoint_path, map_location=map_location)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optim_state") is not None:
        optimizer.load_state_dict(payload["optim_state"])
    if scheduler is not None and payload.get("sched_state") is not None:
        scheduler.load_state_dict(payload["sched_state"])
    return payload

# Backwards-compatible alias
restore = restore_checkpoint

