from __future__ import annotations

from typing import Any, Iterable, Sequence

import os
import torch
from torch.utils.data import DataLoader, Dataset

from seizure_pred.core.config import DataConfig
from seizure_pred.training.registries import DATALOADERS


def _collate_with_meta(batch: Sequence[tuple[torch.Tensor, torch.Tensor, Any]]):
    xs, ys, metas = [], [], []
    for x, y, m in batch:
        xs.append(x)
        ys.append(y)
        metas.append(m)
    return torch.stack(xs, dim=0), torch.stack(ys, dim=0), metas


@DATALOADERS.register("torch", help="Standard torch DataLoader with meta preserved")
def build_torch_dataloader(
    dataset: Dataset,
    cfg: DataConfig,
    *,
    shuffle: bool = True,
    drop_last: bool = False,
    **kwargs,
) -> Iterable:
    """Build a standard torch DataLoader.

    This is the default loader for instance-level training.

    Yielded batch:
        x: (B, C, T)
        y: (B,)
        meta: list[dict]
    """
    # Windows uses 'spawn' start method. Some datasets/transforms are not picklable by default.
    # For a robust default experience, force num_workers=0 unless the user explicitly configured otherwise.
    num_workers = int(getattr(cfg, "num_workers", 0) or 0)
    persistent = bool(getattr(cfg, "persistent_workers", False))
    if os.name == "nt" and num_workers > 0:
        num_workers = 0
        persistent = False

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=persistent and num_workers > 0,
        drop_last=drop_last,
        collate_fn=_collate_with_meta,
        **kwargs,
    )
