# Plugin guide

This project uses a **registry/factory** pattern so adding new datasets, models, losses,
optimizers, schedulers, or dataloaders does not require editing the training loop.

## Add a dataset
Create a module and register a builder:

```python
from torch.utils.data import Dataset
from seizure_pred.training.registries import DATASETS

@DATASETS.register("my_dataset", help="My custom dataset")
def build_dataset(cfg):
    # cfg is usually seizure_pred.core.config.DataConfig or a compatible object
    return MyDataset(root=cfg.dataset_dir, **cfg.kwargs)
```

## Add a dataloader strategy

```python
from seizure_pred.training.registries import DATALOADERS

@DATALOADERS.register("my_loader", help="My custom loader")
def build_loader(dataset, cfg, *, shuffle: bool):
    return MyDataLoader(dataset, batch_size=cfg.batch_size, shuffle=shuffle)
```

## Add a model

```python
import torch.nn as nn
from seizure_pred.training.registries import MODELS

@MODELS.register("my_model")
def build_model(cfg):
    return MyModel(num_classes=cfg.num_classes, **cfg.kwargs)
```

## Add a loss

```python
import torch.nn as nn
from seizure_pred.training.registries import LOSSES

@LOSSES.register("focal")
def build_loss(cfg):
    return FocalLoss(**cfg.kwargs)
```

## Where to import plugins
Registrations happen at import time.

- If you want built-ins, import `seizure_pred.training`.
- If you want custom plugins, import their module before calling CLI/train.

Tip: for user projects, put custom registrations in `my_project/plugins/__init__.py` and import it once.
