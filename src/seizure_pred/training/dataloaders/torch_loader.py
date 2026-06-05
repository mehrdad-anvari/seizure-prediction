from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from seizure_pred.core.config import DataConfig, TrainConfig
from seizure_pred.training.registries import DATALOADERS


def _collate_keep_meta(batch):
    xs, ys, metas = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.as_tensor(ys)
    return x, y, list(metas)


def register_torch_loader():
    @DATALOADERS.register("torch", help="Standard torch DataLoader with meta preserved.")
    def build_torch_loader(ds: Dataset, cfg: TrainConfig | DataConfig, *, shuffle: bool = True) -> DataLoader:
        dc = cfg.data if isinstance(cfg, TrainConfig) else cfg
        return DataLoader(
            ds,
            batch_size=dc.batch_size,
            shuffle=shuffle,
            num_workers=dc.num_workers,
            pin_memory=dc.pin_memory,
            persistent_workers=dc.persistent_workers if dc.num_workers > 0 else False,
            collate_fn=_collate_keep_meta,
        )
