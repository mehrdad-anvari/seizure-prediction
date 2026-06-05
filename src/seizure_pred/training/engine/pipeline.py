from __future__ import annotations

from typing import Iterator, Tuple, Union

from torch.utils.data import Dataset

from seizure_pred.core.config import DataConfig, TrainConfig
from seizure_pred.data.splits import leave_one_out
from seizure_pred.training.registries import DATASETS, DATALOADERS


def _ensure_registries() -> None:
    """Ensure training + model registries are populated.

    Some consumers (e.g., tests) call build_dataset/build_loader directly without
    importing plugin modules first. We keep plugin registration lazy to avoid
    torch-heavy imports for lightweight use-cases, but make this pathway robust.
    """
    from seizure_pred import models as _models
    from seizure_pred import training as _training

    _training.register_all()
    _models.register_all()


def build_dataset(cfg: TrainConfig) -> Dataset:
    """Create dataset from cfg using registry.

    Dataset factory signature should accept a `DataConfig` or `TrainConfig`.
    Preferred: accept `cfg.data` (DataConfig) for clarity.
    """
    if cfg.data.name not in DATASETS:
        _ensure_registries()
    return DATASETS.create(cfg.data.name, cfg.data)


def iter_splits(dataset: Dataset, *, n_folds: int = 5) -> Iterator[Tuple[Dataset, Dataset]]:
    """Yield (train_set, val_set) folds.

    Default uses leave-one-out style splitter implemented in `seizure_pred.data.splits`.
    If you later want other splitters, make this another registry (SPLITTERS).
    """
    yield from leave_one_out(dataset)


def build_loader(name: str, dataset: Dataset, cfg: TrainConfig, *, shuffle: bool) -> object:
    """Build a dataloader using registry.

    DATALOADERS factories should accept:
      (dataset, data_cfg, shuffle=..., **kwargs)
    """
    if name not in DATALOADERS:
        _ensure_registries()
    return DATALOADERS.create(name, dataset, cfg.data, shuffle=shuffle)


def build_dataloader(
    name: str,
    dataset: Dataset,
    cfg: Union[TrainConfig, DataConfig],
    *,
    shuffle: bool,
    **kwargs,
) -> object:
    """Back-compat helper for older scripts.

    Historically some scripts called ``build_dataloader(name, dataset, data_cfg, shuffle=...)``.
    The newer pipeline uses :func:`build_loader` which takes the full TrainConfig.
    This helper accepts either and forwards to the registered dataloader factories.
    """
    data_cfg = cfg.data if isinstance(cfg, TrainConfig) else cfg
    if name not in DATALOADERS:
        _ensure_registries()
    return DATALOADERS.create(name, dataset, data_cfg, shuffle=shuffle, **kwargs)


# Alias (some code imports build_dataloader, others build_loader)
build_dataloader = build_dataloader
