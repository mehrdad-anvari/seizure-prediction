from __future__ import annotations

from dataclasses import dataclass
from typing import Any, DefaultDict, Iterable, List, Tuple

import math
import numpy as np
import torch
from collections import defaultdict

from seizure_pred.core.config import DataConfig
from seizure_pred.training.registries import DATALOADERS


@dataclass
class MilDataLoader:
    """Multiple-instance learning (MIL) bag builder.

    Assumes positive instances have label==1 and are grouped by `dataset.group_ids`.
    Each epoch, positives are packed into bags per group, and negatives are sampled to balance.

    Yields:
        x: (B, bag_size, C, T)
        y: (B,) bag labels
        meta: list[list[meta]]
    """

    dataset: Any
    batch_size: int = 32
    shuffle: bool = True
    bag_size: int = 8
    balance: bool = True
    random_state: int = 0

    def __post_init__(self):
        self.rng = np.random.RandomState(self.random_state)

        self.pos_grouped: DefaultDict[str, List[int]] = defaultdict(list)
        self.neg_indices: List[int] = []
        for i in range(len(self.dataset)):
            if int(self.dataset.y[i]) == 1:
                self.pos_grouped[str(self.dataset.group_ids[i])].append(i)
            else:
                self.neg_indices.append(i)

        self.count_seen = {int(idx): 0 for idx in self.neg_indices}
        self._build_bags()

    def _build_bags(self):
        pos_bags: List[Tuple[np.ndarray, int]] = []
        neg_bags: List[Tuple[np.ndarray, int]] = []

        total_pos_bags = 0
        for group_inds in self.pos_grouped.values():
            inds = np.asarray(group_inds, dtype=int)
            self.rng.shuffle(inds)
            n_bags = len(inds) // self.bag_size
            if n_bags > 0:
                bags = np.array_split(inds[: n_bags * self.bag_size], n_bags)
                pos_bags.extend([(b, 1) for b in bags])
                total_pos_bags += n_bags

        neg_array = np.asarray(list(self.count_seen.keys()), dtype=int)
        seen_counts = np.asarray(list(self.count_seen.values()), dtype=int)
        min_count = seen_counts.min() if len(seen_counts) else 0
        min_mask = seen_counts == min_count
        candidates = neg_array[min_mask]
        self.rng.shuffle(candidates)

        if self.balance:
            n_needed = total_pos_bags * self.bag_size
        else:
            n_needed = len(neg_array)

        if len(candidates) < n_needed:
            remaining = n_needed - len(candidates)
            others = neg_array[~min_mask]
            self.rng.shuffle(others)
            selected = np.concatenate([candidates, others[:remaining]])
        else:
            selected = candidates[:n_needed]

        for idx in selected.tolist():
            self.count_seen[int(idx)] += 1

        n_bags_inter = len(selected) // self.bag_size
        if n_bags_inter > 0:
            bags = np.array_split(selected[: n_bags_inter * self.bag_size], n_bags_inter)
            neg_bags.extend([(b, 0) for b in bags])

        self.all_bags = pos_bags + neg_bags
        if self.shuffle:
            self.rng.shuffle(self.all_bags)

    def __iter__(self):
        self._build_bags()

        for i in range(0, len(self.all_bags), self.batch_size):
            batch = self.all_bags[i : i + self.batch_size]
            batch_data: List[torch.Tensor] = []
            batch_labels: List[int] = []
            batch_metas: List[List[Any]] = []

            for bag_indices, bag_label in batch:
                bag_data: List[torch.Tensor] = []
                bag_metas: List[Any] = []
                instance_labels: List[int] = []

                for idx in bag_indices:
                    x, y, meta = self.dataset[int(idx)]
                    bag_data.append(x)
                    bag_metas.append(meta)
                    instance_labels.append(int(y))

                # sanity: all instance labels should match the bag label
                if any(lbl != bag_label for lbl in instance_labels):
                    raise ValueError("Mixed labels inside a MIL bag")

                batch_data.append(torch.stack(bag_data, dim=0))
                batch_labels.append(bag_label)
                batch_metas.append(bag_metas)

            yield torch.stack(batch_data, dim=0), torch.tensor(batch_labels, dtype=torch.long), batch_metas

    def __len__(self):
        return math.ceil(len(self.all_bags) / self.batch_size)


@DATALOADERS.register("mil", help="MIL bag dataloader (bags built per epoch)")
def build_mil_dataloader(dataset: Any, cfg: DataConfig, *, shuffle: bool = True, **kwargs) -> Iterable:
    return MilDataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        bag_size=int(kwargs.pop("bag_size", 8)),
        balance=bool(kwargs.pop("balance", True)),
        random_state=int(kwargs.pop("random_state", 0)),
    )
