# API training example

This is a lightweight example showing the **library-style** workflow (no CLI) on the built-in **synthetic** dataset.

```python
import os
from dataclasses import asdict
from datetime import datetime

import seizure_pred.models as models
import seizure_pred.training as training

from seizure_pred.core.config import TrainConfig
from seizure_pred.training import MODELS, LOSSES, OPTIMIZERS, SCHEDULERS
from seizure_pred.training.engine.artifacts import ArtifactWriter
from seizure_pred.training.engine.pipeline import build_dataset, iter_splits, build_dataloader
from seizure_pred.training.engine.trainer import Trainer

# Register built-in plugins (models, datasets, loaders, losses, etc.)
training.register_all()
models.register_all()

cfg = TrainConfig()
cfg.task = "prediction"
cfg.device = "cpu"
cfg.epochs = 2
cfg.save_dir = "runs"
cfg.run_name = "api_demo"

# Use synthetic dataset so this example runs without real EEG files.
cfg.data.name = "synthetic"
cfg.data.task = "prediction"
cfg.data.batch_size = 32
cfg.data.num_workers = 0
cfg.data.pin_memory = False
cfg.data.persistent_workers = False
cfg.data.kwargs = {"n": 256, "c": 8, "t": 64, "pos_frac": 0.25, "seed": 1}

# Baseline model
cfg.model.name = "simple_cnn"
cfg.model.in_channels = 8
cfg.model.num_classes = 2

cfg.loss.name = "bce_logits"

cfg.optim.name = "adam"
cfg.optim.lr = 1e-3
cfg.sched.name = None

# Build dataset and take the first split
dataset = build_dataset(cfg)
train_set, val_set = next(iter(iter_splits(dataset, n_folds=2)))

train_loader = build_dataloader("torch", train_set, cfg, shuffle=True)
val_loader = build_dataloader("torch", val_set, cfg, shuffle=False)

# Run directory (mimics CLI layout)
stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
run_dir = os.path.join(cfg.save_dir, cfg.run_name, stamp, "split_0")
os.makedirs(run_dir, exist_ok=True)

writer = ArtifactWriter(run_dir)
writer.write_schema()
writer.write_config(asdict(cfg))

# Build components from registries
model = MODELS.create(cfg.model.name, cfg.model)
loss_fn = LOSSES.create(cfg.loss.name, **(cfg.loss.kwargs or {}))
optimizer = OPTIMIZERS.create(
    cfg.optim.name,
    model.parameters(),
    lr=cfg.optim.lr,
    weight_decay=cfg.optim.weight_decay,
    **(cfg.optim.kwargs or {}),
)

scheduler = None
if cfg.sched.name:
    scheduler = SCHEDULERS.create(cfg.sched.name, optimizer, **(cfg.sched.kwargs or {}))

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    cfg=cfg,
    run_dir=run_dir,
    artifact_writer=writer,
)

best_ckpt = trainer.fit(train_loader=train_loader, val_loader=val_loader)
print("best_checkpoint:", best_ckpt)
print("run_dir:", run_dir)
```
