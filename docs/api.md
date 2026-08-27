# API reference

This page is the practical reference for the parts of `seizure-pred` most users need:
configuration, data splits, datasets, dataloaders, model zoo, losses, optimizers, schedulers, and runnable examples.

## Quick map

- Config: `seizure_pred.core.config.TrainConfig`
- Registry entry points: `seizure_pred.training.registries`
- Training pipeline: `seizure_pred.training.engine.pipeline`
- Trainers: `seizure_pred.training.engine.trainer.Trainer`, `seizure_pred.training.engine.trainer_mil.TrainerMIL`
- Inference: `seizure_pred.inference.predictor.predict`, `seizure_pred.inference.postprocess`, `seizure_pred.inference.calibration`
- Analysis: `seizure_pred.analysis.runner.analyze_run`, `seizure_pred.analysis.summary`, `seizure_pred.analysis.calibration_sweep`
- Benchmark: `seizure_pred.experiments.benchmark`
- XAI: `seizure_pred.inference.xai`

## Configuration

The primary config object is `TrainConfig`, with nested dataclasses for each subsystem.

### `TrainConfig`

- `task`: `"prediction"` or `"detection"`
- `seed`: random seed used for training and split reproducibility
- `device`: `"cuda"` or `"cpu"`
- `epochs`: number of training epochs
- `grad_clip_norm`: gradient clipping value, or `None` to disable
- `amp`: enable automatic mixed precision
- `log_every`: batch logging frequency
- `val_every`: validation frequency in epochs
- `save_dir`: base run directory
- `run_name`: run name prefix
- `monitor` / `monitor_mode`: metric used to keep the best checkpoint
  (`"val_loss"`/`"min"` default; `"auc"`/`"max"` matches the legacy
  best-by-validation-AUC). Also accepts `f1`/`acc`/`precision`/`recall`.
- `data`, `model`, `loss`, `optim`, `sched`, `postprocess`, `callbacks`, `cv`: nested configuration blocks

### `PostprocessConfig`

- `name`: postprocessor registry key (`threshold`, `moving_average`, `hysteresis`, `compose`) or `null`
- `kwargs`: postprocessor-specific options (applied by `predict --apply-postprocess`)

### `CvConfig`

If the `cv` block is configured, the training pipeline runs in **Nested Cross-Validation** mode.
- `outer_method`: `"LOO"` (Leave-One-Out) or `"KFold"`
- `outer_shuffle`: enable/disable shuffling outer folds
- `outer_n_fold`: number of outer folds (for `"KFold"`)
- `outer_mode`: splitting mode for custom KFold (`"per_event_strata"`, `"strata"`, `"split"`, `"random_split"`)
- `outer_M`: number of samples per stratum in custom KFold
- `inner_method`: `"LOO"` or `"KFold"`
- `inner_n_fold`: number of inner folds
- `inner_shuffle`: enable/disable shuffling inner folds
- `inner_mode`: inner custom KFold splitting mode
- `inner_M`: inner custom KFold stratum size
- `random_state`: random seed for splitting reproducibility

### `DataConfig`

- `name`: dataset plugin name, usually `"chbmit_npz"` or `"synthetic"`
- `dataset_dir`: root BIDS or preprocessed dataset path
- `subject_id`: subject identifier, usually `"01"`
- `use_uint16`: load quantized preprocessed NPZ data
- `suffix`: filename suffix for the preprocessed sessions
- `task`: `"prediction"` or `"detection"`
- `batch_size`, `num_workers`, `pin_memory`, `persistent_workers`
- `kwargs`: dataset-specific parameters such as transforms or synthetic dataset knobs

### `ModelConfig`

- `name`: model registry key
- `num_classes`: output class count
- `in_channels`: number of EEG channels
- `sfreq`: sampling frequency when the model needs it
- `kwargs`: model-specific constructor arguments

### `LossConfig`, `OptimConfig`, `SchedConfig`

- `LossConfig.name`: loss registry key, for example `"bce_logits"` or `"mil_bce_logits"`
- `LossConfig.kwargs`: loss-specific options
- `OptimConfig.name`: optimizer registry key, for example `"adam"` or `"adamw"`
- `OptimConfig.lr`, `OptimConfig.weight_decay`, `OptimConfig.kwargs`
- `SchedConfig.name`: scheduler registry key, or `null` for no scheduler
- `SchedConfig.step`: when to step the scheduler, `"epoch"` or `"step"`
- `SchedConfig.kwargs`: scheduler-specific options

## Registry names

The built-in registry containers are:

- `DATASETS`
- `DATALOADERS`
- `MODELS`
- `LOSSES`
- `OPTIMIZERS`
- `SCHEDULERS`
- `EVALUATORS`
- `CALLBACKS`
- `POSTPROCESSORS`

Use `seizure_pred.training.registries.list_all()` to inspect the registered names at runtime.

## Data splits

The repo ships a small splitter module in `seizure_pred.data.splits`.

### `leave_one_out(dataset, shuffle_interictal=False, random_state=0)`

- Outer split for seizure-event style evaluation
- Keeps only samples marked for training when the dataset exposes `is_used_in_train`
- Splits positive event groups one fold at a time and partitions negative windows across folds

### `leave_one_preictal(dataset, method="balanced", shuffle_interictal=False, random_state=0)`

- Alias for the outer CV workflow used by the legacy pipeline
- `method` ∈ {`balanced`, `balanced_shuffled`, `nearest`}: `balanced` partitions
  interictal windows evenly; `balanced_shuffled` randomises them; `nearest`
  picks temporally-nearest interictal windows to each held-out event

### `stratified_kfold(dataset, n_folds=5, shuffle=False, random_state=0)`

- Inner cross-validation split
- Requires `scikit-learn`

### `make_cv_splitter(...)`

- Compatibility helper that dispatches to outer, inner, or custom strata split modes.
- Supported `mode` / `method` values:
  - `"LOO"` / `"leave_one_out"` / `"leave_one_preictal"`: dispatches to preictal leave-one-group-out.
  - `"stratified"`: dispatches to standard scikit-learn `StratifiedKFold`.
  - `"KFold"`: dispatches to custom strata-based chronological `KFold` splitter (using `mode` like `"per_event_strata"`, `"strata"`, etc. and stratum size `M`).

## Datasets

### `chbmit_npz`

Builder: `seizure_pred.training.datasets.chbmit_npz.build_chbmit_npz_dataset`

Core parameters are taken from `DataConfig`:

- `dataset_dir`
- `subject_id`
- `use_uint16`
- `suffix`
- `task`

Additional dataset-specific arguments can be passed through `cfg.kwargs`.
If `online_transforms` or `offline_transforms` is a list of strings, the builder resolves them through the transform registry.

### `synthetic`

Builder: `seizure_pred.training.datasets.synthetic.build_synthetic`

Useful for smoke tests and example runs without EEG files.

- `n`: number of samples
- `c`: number of channels
- `t`: number of time points per sample
- `pos_frac`: positive class fraction
- `seed`: RNG seed
- `task`: copied from `DataConfig.task`

The synthetic dataset exposes `y`, `group_ids`, and `metadata`, which makes it compatible with the splitters and dataloaders.

## Dataloaders

### `torch`

Standard `torch.utils.data.DataLoader` wrapper that preserves the `meta` field in batches.

- Input batch shape: `(B, C, T)`
- Output: `x, y, meta`
- Uses `DataConfig.batch_size`, `num_workers`, `pin_memory`, and `persistent_workers`
- On Windows, worker count is forced to `0` for safety when multiprocessing would be brittle

### `undersample`

Epoch-wise undersampling loader for heavily imbalanced prediction tasks.

- Balances positive windows with sampled negative windows
- Requires dataset labels in `dataset.y`
- Builder argument: `random_state`

### `mil`

Multiple-instance learning loader that groups windows into bags by `dataset.group_ids`.

- Bag tensor shape: `(B, bag_size, C, T)`
- Builder arguments: `bag_size`, `balance`, `random_state`
- Intended for MIL detection workflows

## Model zoo

Model builders consume `ModelConfig` and register under `MODELS`.

### Baseline and common EEG models

- `simple_cnn`: 1D CNN baseline, requires `in_channels`, optional `hidden=64`
- `eegnet`: common EEGNet-style classifier, typical knobs include `num_electrodes`, `chunk_size`, `F1`, `F2`, `D`, `kernel_1`, `kernel_2`, `dropout`
- `eegwavenet` and `eegwavenet_tiny`: WaveNet-style variants, use `model_size="medium"` or `"tiny"`
- `tsception`: temporal-spatial CNN, key knobs include `in_channels`, `chunk_size`, `sampling_rate`, `num_T`, `num_S`, `hidden`, `dropout_rate`
- `fbmsnet`: accepts channel/time parameters and forwards remaining kwargs
- `lmda`, `tslanet`, `cspnet`, `stnet`, `conformer`: wrapper-style builders that inject `in_channels` and `num_classes`
- `simplevit` / `simple_vit`: compact transformer-style EEG model, with patch/grid and head dimensions configurable
- `eegbandclassifier` and `eeg_band_classifier`: band-based classifier variants
- `darnet`: dual attention refinement network, knobs are `chunk_size`, `d_model` (divisible by `num_heads`), `num_heads`, `attn_dropout`
- `mb_dmgc_cwtffnet`: multi-branch model with channel/time and sampling-rate knobs

### Optional models

- `labram`: requires optional dependencies such as `einops`
- `mhanet`: multi-scale hybrid attention network, requires `einops`; `chunk_size` must equal the window length and `num_heads` must divide the channel count

### Paper reconstructions

Rebuilt from papers that publish no reference code; inferred choices are listed in each module docstring.

- `fapex`: fractional neural frame operator + amplitude/phase state-space encoding + linear attention (`patch_size`, `d_model`, `d_state`, `depth`)
- `md_rescapsnet`: CSP → STFT → SE-SA ResNet → capsule routing; reads `sfreq`, accepts `csp_filters`, pairs with the `capsule_margin` loss
- `seizurenet_kan`: PLV graph + KAN-enhanced GCN (`hidden`, `grid_size`, `spline_order`); torch-only
- `seresnet3d`: channel-stacked STFT volume + 3D SE residual modules (`nperseg`, `n_fft`, `stage_channels`); pairs with `focal`
- `sbtm`: spectral/Hjorth/statistical features + Bi-LSTM (`num_steps`, `hidden_size`); the paper's metaheuristic optimiser is not implemented
- `dgcnn2`, `rgnn`: require `torch-geometric`
- `eeg_gnn_ssl`: graph model that may require adjacency helpers and optional scientific packages

The exact constructor parameters are documented inline in each builder module. The most useful discovery pattern is to inspect the corresponding `@MODELS.register(...)` function and the associated `ModelConfig.kwargs` defaults.

## Losses

- `bce_logits`: `pos_weight=None|float|tensor|"auto"`
- `weighted_bce_logits`: BCE with automatic or explicit positive class weighting
- `focal`: `gamma=2.0`, `alpha=0.25`, `reduction="mean"|"sum"|...`
- `preictal_weighted`: `base_loss="bce_logits"|"focal"`, `max_weight=5.0`
- `mil_bce_logits`: `aggregation="max"|"mean"|"logsumexp"`, optional `pos_weight`
- `mil_confident_loss`: MIL loss for instance-level class logits

## Optimizers

- `adam`: `lr=1e-3`, `weight_decay=0.0`
- `adamw`: `lr=1e-3`, `weight_decay=1e-2`
- `sgd`: `lr=1e-2`, `momentum=0.9`, `weight_decay=0.0`, `nesterov=False`

## Schedulers

- `step`: `step_size=10`, `gamma=0.1`
- `cosine`: `T_max`, `eta_min=0.0`
- `onecycle`: `max_lr`, `epochs`, `steps_per_epoch`, plus standard OneCycleLR knobs

## Evaluators and postprocessing

- `binary`: binary metrics (acc/precision/recall/f1/**auc/ap** + confusion)
- `mil_binary`: applies bag aggregation before binary metrics
- Postprocessors: `threshold`, `moving_average`, `hysteresis`, `compose`
- Calibration (`seizure_pred.inference.calibration`): `percentile`, `beta`,
  `isotonic`, `temperature` + `calibrate_ensemble` (analysis-time, fit-on-val)

## Benchmarking & XAI

- `seizure_pred.experiments.benchmark`: params, FLOPs (`thop`), CPU/GPU latency,
  throughput, GPU memory, `torchinfo` summary (see [Benchmarking](benchmark.md)).
- `seizure_pred.inference.xai`: Captum IntegratedGradients attributions
  (optional `captum`; see [XAI](xai.md)).

## Examples

### Synthetic quickstart

```yaml
task: prediction
device: cpu
epochs: 2

data:
  name: synthetic
  task: prediction
  batch_size: 32
  kwargs:
    n: 256
    c: 8
    t: 64
    pos_frac: 0.25
    seed: 1

model:
  name: simple_cnn
  in_channels: 8
  num_classes: 2

loss:
  name: bce_logits

optim:
  name: adam
  lr: 0.001
```

### Minimal training code

```python
from seizure_pred.core.config import TrainConfig
from seizure_pred.training.engine.pipeline import build_dataset, iter_splits, build_loader
from seizure_pred.training.engine.trainer import Trainer
from seizure_pred.training.registries import MODELS, LOSSES, OPTIMIZERS

cfg = TrainConfig()
cfg.data.name = "synthetic"
cfg.data.kwargs = {"n": 256, "c": 8, "t": 64, "pos_frac": 0.25, "seed": 1}
cfg.model.name = "simple_cnn"
cfg.model.in_channels = 8
cfg.epochs = 2

dataset = build_dataset(cfg)
train_set, val_set = next(iter(iter_splits(dataset)))
train_loader = build_loader("torch", train_set, cfg, shuffle=True)
val_loader = build_loader("torch", val_set, cfg, shuffle=False)

model = MODELS.create(cfg.model.name, cfg.model)
loss_fn = LOSSES.create(cfg.loss.name, **(cfg.loss.kwargs or {}))
optimizer = OPTIMIZERS.create(cfg.optim.name, model.parameters(), lr=cfg.optim.lr)
trainer = Trainer(
    model=model,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=None,
    cfg=cfg,
    run_dir="runs/api_example",
)
trainer.fit(train_loader=train_loader, val_loader=val_loader)
```

### Legacy nested CV

For the original nested cross-validation workflow, see `examples/legacy_pipeline.md` and `examples/scripts/legacy_nested_cv.py`.

## Related docs

- `docs/index.md`
- `docs/data_contracts.md`
- `docs/experiments.md`
- `docs/run_artifacts.md`
- `examples/legacy_pipeline.md`
