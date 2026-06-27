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
    
    model_state = None
    if isinstance(payload, dict):
        for k in ["model_state", "model_state_dict", "state_dict", "model"]:
            if k in payload:
                model_state = payload[k]
                break
    if model_state is None:
        model_state = payload
    model.load_state_dict(model_state)

    if optimizer is not None:
        optim_state = None
        if isinstance(payload, dict):
            for k in ["optim_state", "optimizer_state_dict", "optim_state_dict", "optimizer"]:
                if k in payload:
                    optim_state = payload[k]
                    break
        if optim_state is not None:
            optimizer.load_state_dict(optim_state)

    if scheduler is not None:
        sched_state = None
        if isinstance(payload, dict):
            for k in ["sched_state", "scheduler_state_dict", "sched_state_dict", "scheduler"]:
                if k in payload:
                    sched_state = payload[k]
                    break
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)

    return payload

# Backwards-compatible alias
restore = restore_checkpoint

