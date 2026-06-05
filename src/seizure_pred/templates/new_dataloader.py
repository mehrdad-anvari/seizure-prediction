"""Template: add a new dataloader strategy.

Copy to: seizure_pred/training/dataloaders/<your_loader>.py and register into DATALOADERS.
"""

from __future__ import annotations

from torch.utils.data import DataLoader, Dataset

from seizure_pred.core.config import DataConfig
from seizure_pred.training.registries import DATALOADERS


@DATALOADERS.register("my_loader", help="Example dataloader template")
def build_loader(ds: Dataset, cfg: DataConfig, shuffle: bool = True):
    # Must yield batches in the agreed contract:
    # instance: (x[B,C,T], y[B], meta[list[dict]])
    # or MIL:   (x[B,bag,C,T], y[B], meta[list[list[dict]]])
    return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=cfg.num_workers)
