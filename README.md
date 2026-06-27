# seizure-pred

A modular Python library for **seizure prediction** and **seizure detection** workflows on windowed EEG datasets (e.g., CHB-MIT processed to NPZ).
It provides:

- A **pluggable training system** (registries/factories for dataset, dataloader, model, loss, optimizer, scheduler, evaluator, callbacks)
- **Separate inference API** (predict + post-processing)
- **Standard run artifacts** + an **analyze CLI** that produces plots and reports

> This repository is structured as a library under `src/seizure_pred/`.

---

## Install

Editable install (recommended during development):

```bash
pip install -e .
```

Optional extras:

```bash
pip install -e ".[eeg,signal,viz,train]"
```

- `eeg`: preprocessing dependencies (e.g., `mne`)
- `signal`: optional signal/feature dependencies (e.g., `scipy`)
- `viz`: analysis plotting (`matplotlib`)
- `train`: optional training utilities

Graph / GNN models use `torch-geometric` and are kept optional to keep the
base installation simple (especially on CUDA/ARM setups). Install with:

```bash
pip install -e ".[gnn]"
```

---

## Quick start (CLI)

### 1) Preprocess CHB-MIT (BIDS) → NPZ
```bash
seizure-pred preprocess-chbmit \
  --dataset-dir data/BIDS_CHB-MIT \
  --subject 1
```

### 2) Train
```bash
seizure-pred train --config examples/config_prediction.yaml --split-index 0
```

You can choose components by name (registries):

```bash
seizure-pred train --config examples/config_prediction.yaml \
  --override examples/override_onecycle.yaml \
  --strict
```

### 3) Predict
```bash
seizure-pred predict \
  --config examples/config_prediction.yaml \
  # use the printed best_checkpoint from `seizure-pred train`
  --checkpoint runs/<run_name>/<stamp>/split_0/checkpoints/best.pt \
  --split-index 0 \
  --out-dir runs/<run_name>/<stamp>/split_0/eval_split0
```

### 4) Analyze
```bash
seizure-pred analyze --run-dir runs/<run_name>/<stamp>/split_0
```

This writes plots and reports into `runs/<run_name>/<stamp>/split_0/analysis/`.

---

## Quick start (API)

```python
import seizure_pred.training
import seizure_pred.models

from seizure_pred.core.config import TrainConfig
from seizure_pred.training.engine.pipeline import build_dataset, iter_splits, build_loader
from seizure_pred.training.registries import MODELS, LOSSES, OPTIMIZERS, SCHEDULERS
from seizure_pred.training.engine.artifacts import ArtifactWriter
from seizure_pred.training.engine.trainer import Trainer

cfg = TrainConfig()
cfg.data.name = "chbmit_npz"
cfg.data.subject_id = "01"
cfg.task = "prediction"
cfg.model.name = "simple_cnn"
cfg.loss.name = "bce_logits"
cfg.optim.name = "adam"

seizure_pred.training.register_all()  # ensure registries are populated

dataset = build_dataset(cfg)
train_ds, val_ds = next(iter(iter_splits(dataset)))

train_loader = build_loader("torch", train_ds, cfg, shuffle=True)
val_loader = build_loader("torch", val_ds, cfg, shuffle=False)

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

writer = ArtifactWriter("runs/api_example")
writer.write_schema()
writer.write_config(cfg)

trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    cfg=cfg,
    run_dir="runs/api_example",
    artifact_writer=writer,
)
trainer.fit(train_loader=train_loader, val_loader=val_loader)
```

For a fuller reference, see `docs/api.md`. It includes:

- config fields and defaults
- built-in datasets and dataloaders
- data splitters for prediction and detection workflows
- model zoo parameters and optional dependencies
- loss, optimizer, scheduler, evaluator, and postprocess knobs
- copyable training examples

---

## Configuration

- Configs are YAML/JSON mapped into dataclasses (`TrainConfig`).
- Unknown keys / wrong types are rejected (validator).

See:
- `examples/config_prediction.yaml`
- `examples/config_mil_detection.yaml`

---

## Transforms

Signal and feature transforms live under `src/seizure_pred/transforms/`.

Use them directly, or via the small factory in `seizure_pred.transforms.registry`.
Some transforms require optional extras (`.[signal]` and/or `.[eeg]`).

---

## Extending the library

Most extension points are **registries**. You add a new component by implementing a factory and registering it.

- Dataset: `DATASETS.register("name")`
- Dataloader: `DATALOADERS.register("name")`
- Model: `MODELS.register("name")`
- Loss: `LOSSES.register("name")`
- Optimizer: `OPTIMIZERS.register("name")`
- Scheduler: `SCHEDULERS.register("name")`
- Evaluator: `EVALUATORS.register("name")`
- Callback: `CALLBACKS.register("name")`
- Postprocess: `POSTPROCESSORS.register("name")`

See `docs/extending.md` and templates in `src/seizure_pred/templates/`.

---

## Run artifacts schema (stable)

Each run directory contains:

- `schema.json` (schema version + minimal metadata)
- `config.json` (full resolved config snapshot)
- `history.jsonl` (epoch logs)
- `metrics.json` (best-val metrics)
- `predictions.jsonl` (optional; saved by training/predict)

See `docs/run_artifacts.md`.

---

## Migration from the old repo

See `MIGRATION.md`.


## Release

To cut a release, follow `RELEASE_CHECKLIST.md`. Recommended tag format: `vX.Y.Z` (e.g., `v0.2.0`).
