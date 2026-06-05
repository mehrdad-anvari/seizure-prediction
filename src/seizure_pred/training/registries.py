from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset

from seizure_pred.core.registry import Registry

# Core build targets
DATASETS: Registry[Dataset] = Registry("dataset")
DATALOADERS: Registry[object] = Registry("dataloader")  # factories return DataLoader-like objects
MODELS: Registry[nn.Module] = Registry("model")
LOSSES: Registry[nn.Module] = Registry("loss")
OPTIMIZERS: Registry[torch.optim.Optimizer] = Registry("optimizer")
SCHEDULERS: Registry[object] = Registry("scheduler")  # schedulers vary in base type

# Extensibility points
EVALUATORS: Registry[object] = Registry("evaluator")      # callable / object with evaluate(...)
CALLBACKS: Registry[object] = Registry("callback")        # training hooks
POSTPROCESSORS: Registry[object] = Registry("postprocess")  # inference postproc

def list_all() -> dict[str, list[str]]:
    return {
        "datasets": sorted(list(DATASETS.names())),
        "dataloaders": sorted(list(DATALOADERS.names())),
        "models": sorted(list(MODELS.names())),
        "losses": sorted(list(LOSSES.names())),
        "optimizers": sorted(list(OPTIMIZERS.names())),
        "schedulers": sorted(list(SCHEDULERS.names())),
        "evaluators": sorted(list(EVALUATORS.names())),
        "callbacks": sorted(list(CALLBACKS.names())),
        "postprocessors": sorted(list(POSTPROCESSORS.names())),
    }