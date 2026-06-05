from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from seizure_pred.core.config import DataConfig, TrainConfig
from seizure_pred.training.registries import DATALOADERS
from .torch_loader import _collate_keep_meta


class BalancedBinarySampler(Sampler[int]):
    """Sampler that undersamples the majority class to match minority class each epoch."""

    def __init__(self, labels: Sequence[int], *, seed: int = 0):
        self.labels = np.asarray(labels, dtype=int)
        self.seed = int(seed)

        self.pos_idx = np.where(self.labels == 1)[0]
        self.neg_idx = np.where(self.labels == 0)[0]
        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            # fall back to all indices
            self.pos_idx = np.arange(len(self.labels))
            self.neg_idx = np.arange(len(self.labels))

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        n = min(len(self.pos_idx), len(self.neg_idx))
        pos = rng.choice(self.pos_idx, size=n, replace=False)
        neg = rng.choice(self.neg_idx, size=n, replace=False)
        idx = np.concatenate([pos, neg])
        rng.shuffle(idx)
        return iter(idx.tolist())

    def __len__(self) -> int:
        return 2 * min(len(self.pos_idx), len(self.neg_idx))


def _extract_labels(ds: Dataset) -> List[int]:
    # expects dataset[i] -> (x, y, meta)
    ys = []
    for i in range(len(ds)):
        _, y, _ = ds[i]
        ys.append(int(y))
    return ys


def register_undersample_loader():
    @DATALOADERS.register("undersample", help="Torch DataLoader + per-epoch undersampling for binary labels.")
    def build_undersample_loader(ds: Dataset, cfg: TrainConfig | DataConfig, *, shuffle: bool = True) -> DataLoader:
        dc = cfg.data if isinstance(cfg, TrainConfig) else cfg
        labels = _extract_labels(ds)
        sampler = BalancedBinarySampler(labels, seed=getattr(cfg, "seed", 0))
        return DataLoader(
            ds,
            batch_size=dc.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=dc.num_workers,
            pin_memory=dc.pin_memory,
            persistent_workers=dc.persistent_workers if dc.num_workers > 0 else False,
            collate_fn=_collate_keep_meta,
        )
