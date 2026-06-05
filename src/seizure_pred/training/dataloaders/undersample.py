from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List

import numpy as np
import torch

from seizure_pred.core.config import DataConfig
from seizure_pred.training.registries import DATALOADERS


@dataclass
class UnderSampledDataLoader:
    """Undersample negative class (label==0) each epoch to match positives.

    Intended for highly imbalanced *prediction* training.

    Requirements on dataset:
        - dataset.y: 1D array-like of 0/1 labels (available on SubsetWithInfo)
        - dataset.__getitem__(i) -> (x, y, meta)

    Yields:
        x: (B,C,T)
        y: (B,)
        meta: list
    """

    dataset: Any
    batch_size: int = 32
    shuffle: bool = True
    random_state: int = 0

    def __post_init__(self):
        self.rng = np.random.RandomState(self.random_state)

        pos, neg = [], []
        for i in range(len(self.dataset)):
            if int(self.dataset.y[i]) == 1:
                pos.append(i)
            else:
                neg.append(i)
        self.pos_indices = np.asarray(pos, dtype=int)
        self.neg_indices = np.asarray(neg, dtype=int)
        self.count_seen = {int(idx): 0 for idx in self.neg_indices.tolist()}
        self.all_indices = self._sample_epoch_indices()

    def _sample_epoch_indices(self) -> np.ndarray:
        n_pos = len(self.pos_indices)
        n_neg = len(self.neg_indices)

        if n_pos == 0:
            # Degenerate: just return negatives
            selected_neg = self.neg_indices
        elif n_neg > n_pos:
            inter_array = np.array(list(self.count_seen.keys()), dtype=int)
            seen_counts = np.array(list(self.count_seen.values()), dtype=int)

            min_count = seen_counts.min()
            min_mask = seen_counts == min_count
            min_indices = inter_array[min_mask]

            if len(min_indices) < n_pos:
                remaining = n_pos - len(min_indices)
                not_min = inter_array[~min_mask]
                selected_extra = self.rng.choice(not_min, size=remaining, replace=False)
                selected_neg = np.concatenate([min_indices, selected_extra])
            else:
                selected_neg = self.rng.choice(min_indices, size=n_pos, replace=False)
        else:
            selected_neg = self.neg_indices

        for idx in selected_neg:
            self.count_seen[int(idx)] += 1

        all_idx = np.concatenate([self.pos_indices, selected_neg]).astype(int)
        if self.shuffle:
            self.rng.shuffle(all_idx)
        return all_idx

    def __iter__(self):
        self.all_indices = self._sample_epoch_indices()

        for i in range(0, len(self.all_indices), self.batch_size):
            batch_indices = self.all_indices[i : i + self.batch_size]
            batch_x: List[torch.Tensor] = []
            batch_y: List[torch.Tensor] = []
            batch_meta: List[Any] = []
            for idx in batch_indices:
                x, y, meta = self.dataset[int(idx)]
                batch_x.append(x)
                batch_y.append(y)
                batch_meta.append(meta)
            yield torch.stack(batch_x), torch.stack(batch_y), batch_meta

    def __len__(self):
        return (len(self.all_indices) + self.batch_size - 1) // self.batch_size


@DATALOADERS.register("undersample", help="Epoch-wise undersampling loader (pos + matched neg)")
def build_undersample_dataloader(dataset: Any, cfg: DataConfig, *, shuffle: bool = True, **kwargs) -> Iterable:
    return UnderSampledDataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        random_state=kwargs.pop("random_state", 0),
    )
