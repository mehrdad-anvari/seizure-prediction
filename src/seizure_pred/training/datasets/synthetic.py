from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from seizure_pred.core.config import DataConfig
from seizure_pred.training.registries import DATASETS


class SyntheticEEGDataset(Dataset):
    """Tiny synthetic dataset for smoke tests and quick experiments.

    Provides CHBMIT-like attributes needed by splitters:
      - y
      - group_ids
      - metadata

    Produces:
      x: (C, T) float32
      y: int64 0/1
      meta: dict
    """

    def __init__(
        self,
        n: int = 256,
        c: int = 8,
        t: int = 64,
        pos_frac: float = 0.25,
        seed: int = 1,
        task: str = "prediction",
    ):
        self.n = int(n)
        self.c = int(c)
        self.t = int(t)
        self.pos_frac = float(pos_frac)
        self.seed = int(seed)
        self.task = str(task)

        rng = np.random.default_rng(self.seed)
        self.y = (rng.random(self.n) < self.pos_frac).astype(np.int64)

        # Two groups so leave_one_out yields non-empty train/test splits.
        self.group_ids = (np.arange(self.n) % 2).astype(np.int64)

        # Create signals with a faint class-dependent mean shift.
        x = rng.standard_normal((self.n, self.c, self.t)).astype(np.float32)
        x[self.y == 1] += 0.5
        self.x = x

        self.metadata = [
            {"index": int(i), "group": int(self.group_ids[i]), "task": self.task} for i in range(self.n)
        ]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        x = torch.from_numpy(self.x[idx])
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        meta = dict(self.metadata[idx])
        return x, y, meta


@DATASETS.register("synthetic", help="Synthetic EEG-like dataset for tests (no real files).")
def build_synthetic(cfg: DataConfig) -> Dataset:
    kw = dict(cfg.kwargs or {})
    return SyntheticEEGDataset(
        n=kw.get("n", 256),
        c=kw.get("c", 8),
        t=kw.get("t", 64),
        pos_frac=kw.get("pos_frac", 0.25),
        seed=kw.get("seed", 1),
        task=getattr(cfg, "task", "prediction"),
    )
