from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from seizure_pred.core.config import DataConfig, TrainConfig
from seizure_pred.training.registries import DATALOADERS


def _collate_mil(batch):
    # batch: list[(bag_x, y, bag_meta)]
    xs, ys, metas = zip(*batch)
    x = torch.stack(xs, dim=0)  # (B,bag,C,T)
    y = torch.as_tensor(ys)
    return x, y, list(metas)


class MilBagDataset(Dataset):
    """Wrap an instance dataset (x,y,meta) into bags.

    Each item becomes a bag of `bag_size` instances sampled from the same class.
    This is a simple baseline; users can implement smarter bagging and register it.
    """
    def __init__(self, base: Dataset, bag_size: int = 8, seed: int = 0):
        self.base = base
        self.bag_size = int(bag_size)
        self.seed = int(seed)

        # precompute indices per class
        ys = [int(base[i][1]) for i in range(len(base))]
        ys = np.asarray(ys, dtype=int)
        self.pos = np.where(ys == 1)[0]
        self.neg = np.where(ys == 0)[0]
        if len(self.pos) == 0 or len(self.neg) == 0:
            self.pos = np.arange(len(base))
            self.neg = np.arange(len(base))

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        _, y, _ = self.base[idx]
        y = int(y)
        rng = np.random.default_rng(self.seed + idx)
        pool = self.pos if y == 1 else self.neg
        chosen = rng.choice(pool, size=self.bag_size, replace=len(pool) < self.bag_size)

        xs: List[torch.Tensor] = []
        metas: List[dict] = []
        for j in chosen:
            xj, _, mj = self.base[int(j)]
            xs.append(xj)
            metas.append(mj)
        bag_x = torch.stack(xs, dim=0)  # (bag,C,T)
        return bag_x, y, metas


def register_mil_loader():
    @DATALOADERS.register("mil", help="MIL DataLoader: wraps instance dataset into simple same-label bags.")
    def build_mil_loader(ds: Dataset, cfg: TrainConfig | DataConfig, *, shuffle: bool = True) -> DataLoader:
        dc = cfg.data if isinstance(cfg, TrainConfig) else cfg
        bag_size = int(getattr(dc, "kwargs", {}).get("bag_size", 8))
        base_seed = int(getattr(cfg, "seed", 0)) if isinstance(cfg, TrainConfig) else 0
        mil_ds = MilBagDataset(ds, bag_size=bag_size, seed=base_seed)
        return DataLoader(
            mil_ds,
            batch_size=dc.batch_size,
            shuffle=shuffle,
            num_workers=dc.num_workers,
            pin_memory=dc.pin_memory,
            persistent_workers=dc.persistent_workers if dc.num_workers > 0 else False,
            collate_fn=_collate_mil,
        )
