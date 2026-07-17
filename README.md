# seizure-pred

A modular Python library for **seizure prediction** and **seizure detection** on
windowed EEG (built around the CHB-MIT dataset processed to NPZ). It is a
refactor of the older `CP-DMGC-CWT` research codebase into a clean,
plugin-based library (`src/seizure_pred/`) with a stable CLI, a typed config,
nested cross-validation, calibration-aware analysis, benchmarking and XAI.

## What it gives you

- **Pluggable training system** — registries/factories for *dataset, dataloader,
  model, loss, optimizer, scheduler, evaluator, callback, postprocessor*.
- **Two tasks** — `prediction` (preictal vs interictal) and `detection`
  (seizure vs background), with task-aware label mapping.
- **Nested cross-validation** — outer × inner loops with leave-one-preictal-out
  (`balanced` / `balanced_shuffled` / `nearest`) and custom stratified K-Fold
  (`random_split` / `split` / `strata` / `per_event_strata`).
- **Class-imbalance handling** — undersampling loader, MIL bags, `pos_weight`
  auto-estimation, focal loss, and temporal (preictal-position) weighting.
- **Separate inference API** — `predict` / `predict_ensemble` with streaming
  post-processing (threshold, moving-average, hysteresis, compose).
- **Probability calibration** — `percentile`, `beta`, `isotonic`,
  `temperature` (fit-on-validation, AUC-weighted ensemble).
- **Analysis CLI** — per-split reports/curves plus a
  `calibration × MA-window × threshold` variant sweep, Pareto frontier,
  suppression-based FPR/hour, and best-variant CSVs.
- **Benchmarking** — params, FLOPs/MACs (`thop`), CPU/GPU latency, throughput,
  GPU memory (`torchinfo`).
- **Explainability (XAI)** — Captum IntegratedGradients channel attributions
  and topomaps (optional).
- **Standard run artifacts** — `schema.json`, `config.json`, `history.jsonl`,
  `metrics.json`, `predictions.jsonl`, `checkpoints/`.
- **22 models** — from `simple_cnn`, `eegnet`, `eegwavenet` to graph models
  (`rgnn`, `dgcnn2`, `eeg_gnn_ssl`) and the flagship `mb_dmgc_cwtffnet`.

> The library lives under `src/seizure_pred/`. Install it editable and import
> `seizure_pred`.

---

## Install

Editable install (recommended during development):

```bash
pip install -e .
```

Optional extras (the library degrades gracefully when these are missing):

```bash
pip install -e ".[train,viz,eeg,signal,gnn]"
```

| Extra | Adds |
|-------|------|
| `train` | `pyyaml`, `scikit-learn` (configs + stratified CV) |
| `viz` | `matplotlib` (analysis plots) |
| `eeg` | `mne`, `mne-connectivity` (preprocessing + connectivity features) |
| `signal` | `scipy` (signal/feature transforms, calibration optimisation) |
| `gnn` | `torch-geometric` (graph models: `rgnn`, `dgcnn2`) |

Additional optional packages (not in extras, install on demand):
`captum` (XAI), `thop` + `torchinfo` (benchmark FLOPs/summary), `seaborn`
(nicer analysis plots).

---

## Quick start (CLI)

### 1) Preprocess CHB-MIT (BIDS) → NPZ
```bash
seizure-pred preprocess-chbmit --dataset-dir data/BIDS_CHB-MIT --subject 1
```

### 2) Train (single-level CV)
```bash
seizure-pred train --config examples/config_prediction.yaml
```

### 3) Train with nested cross-validation
```bash
seizure-pred train --config examples/config_nested_cv.yaml
```
This writes `raw_predictions.pkl` plus per-split ensembled predictions.

### 4) Predict
```bash
seizure-pred predict --config examples/config_prediction.yaml \
  --checkpoint runs/<run_name>/<stamp>
```
Pass `--apply-postprocess` to apply the configured `postprocess` block.

### 5) Analyze
```bash
seizure-pred analyze --run-dir runs/<run_name>/<stamp>
```
For nested-CV runs this also runs the **calibration sweep**
(`--calibration-methods none percentile beta isotonic temperature
--ma-windows 1 3 5 7 10 --thresholds 0.3 0.4 0.5 0.6 0.7
--percentiles 5 10 15 20 --suppression-duration 60`).

### 6) Benchmark models
```bash
seizure-pred benchmark --models eegnet eegwavenet mb_dmgc_cwtffnet --batch-sizes 1 32
```

### 7) List registered plugins
```bash
seizure-pred list
```

You can override config values from the CLI:
```bash
seizure-pred train --config examples/config_prediction.yaml \
  --override examples/override_onecycle.yaml --strict
```

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
cfg.monitor = "auc"          # select best epoch by validation AUC
cfg.monitor_mode = "max"

seizure_pred.training.register_all()   # populate registries
seizure_pred.models.register_all()

dataset = build_dataset(cfg)
train_ds, val_ds = next(iter(iter_splits(dataset, cfg.data)))

train_loader = build_loader("torch", train_ds, cfg, shuffle=True)
val_loader = build_loader("torch", val_ds, cfg, shuffle=False)

model = MODELS.create(cfg.model.name, cfg.model)
loss_fn = LOSSES.create(cfg.loss.name, **(cfg.loss.kwargs or {}))
optimizer = OPTIMIZERS.create(cfg.optim.name, model.parameters(),
                              lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay,
                              **(cfg.optim.kwargs or {}))
scheduler = None
if cfg.sched.name:
    scheduler = SCHEDULERS.create(cfg.sched.name, optimizer, **(cfg.sched.kwargs or {}))

writer = ArtifactWriter("runs/api_example")
trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer,
                  scheduler=scheduler, cfg=cfg, run_dir="runs/api_example",
                  artifact_writer=writer)
trainer.fit(train_loader=train_loader, val_loader=val_loader)
```

For the full reference see `docs/api.md`.

---

## Configuration

Configs are YAML/JSON mapped into dataclasses (`TrainConfig`). The validator
resolves (possibly string) annotations and **rejects unknown keys and wrong
types** with clear errors. Top-level fields:

- `task`, `seed`, `device`, `epochs`, `grad_clip_norm`, `amp`, `log_every`,
  `val_every`, `save_dir`, `run_name`
- `monitor` / `monitor_mode` — metric used to keep the best checkpoint
  (`val_loss`/`min` default; use `auc`/`max` to match the legacy
  "best by validation AUC").
- `data`, `model`, `loss`, `optim`, `sched`, `postprocess`, `callbacks`, `cv`

See `examples/config_prediction.yaml`, `examples/config_nested_cv.yaml`,
`examples/config_mil_detection.yaml`, `examples/config_synthetic.yaml`.

---

## Cross-validation

Splitters live in `seizure_pred.data.splits`.

- **Leave-one-preictal-out** (`LOO`): holds out one positive (seizure) event per
  fold. `method` ∈ {`balanced`, `balanced_shuffled`, `nearest`}:
  - `balanced` — interictal windows partitioned evenly (chronological).
  - `balanced_shuffled` — balanced + randomised interictal selection.
  - `nearest` — test interictal windows are the *temporally nearest* to the
    held-out event (reduces train/test distribution mismatch).
- **Stratified K-Fold** (`stratified`): scikit-learn `StratifiedKFold` (inner CV).
- **Custom K-Fold** (`KFold`): chronological, with modes
  `random_split` / `split` / `strata` / `per_event_strata` and a stratum size
  `M` that limits leakage from overlapping windows.

When a `cv:` block is present, `seizure-pred train` runs the full
**outer × inner nested CV**, writes `raw_predictions.pkl`, and AUC-weight
ensembles the inner folds' test predictions per outer split.

---

## Losses, optimizers, schedulers

| Losses | Optimizers | Schedulers |
|--------|-----------|-----------|
| `bce_logits` (`pos_weight`/"auto") | `adam` | `step` |
| `weighted_bce_logits` | `adamw` | `cosine` |
| `focal` (`gamma`, `alpha`) | `sgd` (momentum/nesterov) | `onecycle` |
| `preictal_weighted` (temporal) | | |
| `mil_bce_logits` (`aggregation`) | | |
| `mil_confident_loss` | | |

---

## Dataloaders

- `torch` — standard `DataLoader` preserving per-sample `meta`.
- `undersample` — epoch-wise undersampling of the majority (interictal) class,
  preferring least-seen windows.
- `mil` — builds MIL bags per epoch grouped by `group_ids`
  (`(B, bag_size, C, T)`), with optional balancing.

---

## Post-processing & calibration

Inference postprocessors (`POSTPROCESSORS`): `threshold`, `moving_average`,
`hysteresis` (dual-threshold + min on/off durations), `compose`.

Probability calibration (`seizure_pred.inference.calibration`):
`ProbabilityCalibrator(method=...)` with `percentile` / `beta` / `isotonic` /
`temperature`, plus `calibrate_ensemble(...)` for AUC-weighted ensembling.
Calibration is fit on validation probabilities and applied to test
probabilities; the `analyze` CLI runs the full variant sweep automatically for
nested-CV runs.

---

## Analysis outputs

For each split: `analysis/report.json`, `report.txt`, `roc.png`, `pr.png`,
`confusion.png`, `history.png`. Aggregated under `<run>/analysis/`:

- `analysis_summary.json`, `pareto_optimal_variants.csv` (MA × threshold sweep)
- For nested CV: `variant_summary.csv`, `best_variants_{auc,sensitivity,f1,
  fpr_per_hour}.csv`, `calibration_comparison.csv`, `ma_window_comparison.csv`,
  `threshold_comparison.csv`, `pareto_optimal_variants.csv`,
  `calibration_summary.json`, and plots under `analysis/plots/`.

Clinical metrics: AUC, average precision, sensitivity (≥1 preictal detected per
event), FPR/hour (interictal-only), **FPR/hour with suppression window**,
specificity, time-to-detection, confusion counts.

---

## Transforms

Signal and feature transforms live under `src/seizure_pred/transforms/`. Use
them directly or via `seizure_pred.transforms.registry.create_transform(name)`.
Some require optional extras (`.[signal]`, `.[eeg]`).

- **Signal**: `instance_norm`, `to_grid`, `filterbank`, `wavelet_filterbank`.
- **Feature** (35+): basic stats (mean/std/skew/kurt/RMS/line-length/ZCR/Hjorth),
  band powers (δ/θ/α/β/γ), spectral summaries (entropy/iHMF/SEF95/peak),
  connectivity (coherence/PLV/ImCoh/PLI/WPLI, mean-abs-correlation),
  differential entropy.

---

## Extending the library

Most extension points are **registries** — implement a factory and register it:

- `DATASETS.register("name")`, `DATALOADERS.register("name")`
- `MODELS.register("name")`, `LOSSES.register("name")`
- `OPTIMIZERS.register("name")`, `SCHEDULERS.register("name")`
- `EVALUATORS.register("name")`, `CALLBACKS.register("name")`,
  `POSTPROCESSORS.register("name")`

See `docs/extending.md` and templates in `src/seizure_pred/templates/`.

---

## Run artifacts schema (stable)

Each run directory contains:

- `schema.json` (schema version + minimal metadata)
- `config.json` (full resolved config snapshot)
- `history.jsonl` (epoch logs, incl. `val_auc`/`val_ap`)
- `metrics.json` (best-val metrics)
- `predictions.jsonl` (per-window `y_true`/`logit`/`prob`/`y_pred`/`meta`)
- `checkpoints/best.pt` (and `last.pt` fallback)

See `docs/run_artifacts.md`.

---

## Migration from the old repo

See `MIGRATION.md` and `docs/migration.md`. The legacy training entrypoints are
retained under `src/seizure_pred/legacy/`.

## Release

To cut a release, follow `RELEASE_CHECKLIST.md`. Recommended tag format:
`vX.Y.Z` (e.g., `v0.2.0`).
