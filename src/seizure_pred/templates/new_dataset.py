"""Template: add a new dataset.

Copy to: seizure_pred/training/datasets/<your_dataset>.py
and implement the builder registered into DATASETS.
"""

from __future__ import annotations

from torch.utils.data import Dataset

from seizure_pred.core.config import DataConfig
from seizure_pred.training.registries import DATASETS


class MyDataset(Dataset):
    def __init__(self, root: str, **kwargs):
        self.root = root
        self.kwargs = kwargs

    def __len__(self) -> int:
        return 0

    def __getitem__(self, idx: int):
        # MUST return: (x, y, meta)
        # x: Tensor [C,T] or [C,T,...] depending on your model
        # y: int or float label
        # meta: dict (must be JSON-serializable if you want predictions logged)
        raise NotImplementedError


@DATASETS.register("my_dataset", help="Example dataset template")
def build_my_dataset(cfg: DataConfig) -> Dataset:
    return MyDataset(root=cfg.dataset_dir, **cfg.kwargs)
