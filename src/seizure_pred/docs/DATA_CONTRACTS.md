# Data contracts

This library standardizes dataset/dataloader outputs so new components can be added without
changing the training loop.

## Dataset `__getitem__` contract
A dataset **must** return a 3-tuple:

1) `x`: `torch.Tensor` (e.g. `(C, T)`)
2) `y`: `torch.Tensor` (e.g. a scalar class index, or a multi-label vector)
3) `meta`: `dict` of metadata (event id, start time, subject, raw label, etc.)

## Instance dataloader contract
An instance dataloader yields batches:

- `x`: `(B, C, T)`
- `y`: `(B,)` or `(B, K)` depending on task
- `meta`: list of `dict` with length `B`

## MIL dataloader contract
MIL loaders yield bags:

- `x`: `(B, bag, C, T)`
- `y`: `(B,)`
- `meta`: list of `list[dict]` with length `B`, each inner list length `bag`

## CHB-MIT labels
### Prediction task
- Positive class: `preictal`
- Training negatives: `interictal`

### Detection task
- Positive class: `seizure`
- Training negatives: `interictal`, `preictal`, `pre_buffer`, `post_buffer`


4) Smoke test (end-to-end)

Create: `tests/test_smoke_pipeline.py`

> This assumes your earlier chunks already include:
> - `DATASETS["synthetic"]` (or any tiny dataset plugin you have)
> - `DATALOADERS["torch"]`
> - `MODELS["simple_cnn"]`
> - `Trainer`, `ArtifactWriter`, `predict()`, `analyze_run()`

```python
import os
import tempfile

import seizure_pred.training  # noqa: F401 (register plugins)
import seizure_pred.models    # noqa: F401

from seizure_pred.core.config import TrainConfig
from seizure_pred.training.engine.pipeline import build_dataset, iter_splits, build_loader
from seizure_pred.training.engine.trainer import Trainer
from seizure_pred.training.engine.artifacts import ArtifactWriter
from seizure_pred.inference.predictor import predict
from seizure_pred.analysis.runner import analyze_run


def test_smoke_end_to_end():
    cfg = TrainConfig()
    cfg.device = "cpu"
    cfg.epochs = 2

    # Use synthetic dataset plugin to avoid real EEG files
    cfg.data.name = "synthetic"
    cfg.data.batch_size = 16
    cfg.data.num_workers = 0
    cfg.data.pin_memory = False
    cfg.data.persistent_workers = False
    cfg.data.kwargs = {"n": 128, "c": 8, "t": 64, "pos_frac": 0.25, "seed": 1}

    cfg.model.name = "simple_cnn"
    cfg.model.num_classes = 1
    cfg.model.kwargs = {"in_channels": 8}

    ds = build_dataset(cfg)
    train_set, val_set = list(iter_splits(ds, n_folds=3))[0]
    train_loader = build_loader("torch", train_set, cfg, shuffle=True)
    val_loader = build_loader("torch", val_set, cfg, shuffle=False)

    with tempfile.TemporaryDirectory() as td:
        trainer = Trainer(cfg, run_dir=td)
        best = trainer.fit(train_loader, val_loader)
        assert os.path.exists(best)

        out_dir = os.path.join(td, "eval")
        writer = ArtifactWriter(out_dir)
        writer.write_schema()

        rows = list(predict(trainer.model, val_loader, device="cpu", is_mil=False))
        writer.write_predictions(rows, subdir="predictions")

        report = analyze_run(out_dir, threshold=0.5)
        assert "f1" in report