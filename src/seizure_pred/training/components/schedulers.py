from __future__ import annotations

from typing import Optional

import torch

from seizure_pred.training.registries import SCHEDULERS


@SCHEDULERS.register("step", help="StepLR")
def build_step(optimizer: torch.optim.Optimizer, step_size: int = 10, gamma: float = 0.1, **kwargs):
    _ = kwargs
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(step_size), gamma=float(gamma))


@SCHEDULERS.register("cosine", help="CosineAnnealingLR")
def build_cosine(optimizer: torch.optim.Optimizer, T_max: int, eta_min: float = 0.0, **kwargs):
    _ = kwargs
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(T_max), eta_min=float(eta_min))


@SCHEDULERS.register("onecycle", help="OneCycleLR (requires steps_per_epoch and epochs)")
def build_onecycle(
    optimizer: torch.optim.Optimizer,
    max_lr: float,
    epochs: int,
    steps_per_epoch: int,
    pct_start: float = 0.3,
    anneal_strategy: str = "cos",
    div_factor: float = 25.0,
    final_div_factor: float = 1e4,
    three_phase: bool = False,
    **kwargs,
):
    _ = kwargs
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=float(max_lr),
        epochs=int(epochs),
        steps_per_epoch=int(steps_per_epoch),
        pct_start=float(pct_start),
        anneal_strategy=str(anneal_strategy),
        div_factor=float(div_factor),
        final_div_factor=float(final_div_factor),
        three_phase=bool(three_phase),
    )
