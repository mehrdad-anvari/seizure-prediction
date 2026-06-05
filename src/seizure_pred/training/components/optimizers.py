from __future__ import annotations

import torch

from seizure_pred.training.registries import OPTIMIZERS


@OPTIMIZERS.register("adam")
def build_adam(params, lr: float = 1e-3, weight_decay: float = 0.0, **kwargs):
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZERS.register("adamw")
def build_adamw(params, lr: float = 1e-3, weight_decay: float = 1e-2, **kwargs):
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)


@OPTIMIZERS.register("sgd")
def build_sgd(
    params,
    lr: float = 1e-2,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    nesterov: bool = False,
    **kwargs,
):
    return torch.optim.SGD(
        params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
        **kwargs,
    )
