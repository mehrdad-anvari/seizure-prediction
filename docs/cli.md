# Command-line interface

The `seizure-pred` CLI is installed by `pip install -e .` (entry point
`seizure-pred = seizure_pred.cli.main:main`). Heavy modules are imported lazily,
so `--help` is fast.

## `seizure-pred list`

Print all registered plugins (datasets, dataloaders, models, losses,
optimizers, schedulers, evaluators, callbacks, postprocessors).

## `seizure-pred preprocess-chbmit`

Preprocess raw CHB-MIT BIDS EDF + events into per-session NPZ files.

Common flags: `--dataset-dir`, `--subject` (`1,2,3` or `all`),
`--save-uint16`, `--no-filter`/`--no-ica`/`--no-downsample`,
`--filter-type {FIR,IIR}`, `--l-freq`, `--h-freq`, `--sfreq-new`,
`--downsample-method {polyphase,fft,resample}`, `--normalize {zscore,robust}`,
`--preictal-minutes`, `--pre-buffer-minutes`, `--post-buffer-minutes`,
`--segment-sec`, `--preictal-oversample-factor`, `--seizure-oversample-factor`,
`--plot`, `--plot-psd`.

## `seizure-pred train`

```bash
seizure-pred train --config examples/config_prediction.yaml
```

Flags: `--config`, `--override`, `--n-folds`, `--dataloader`, `--mil`,
`--strict`, `--print-config`, and nested-CV overrides `--outer-method`,
`--inner-method`, `--outer-n-fold`, `--inner-n-fold`,
`--outer-shuffle`/`--no-outer-shuffle`, `--inner-shuffle`/`--no-inner-shuffle`.

If the config contains a `cv:` block (or any `--outer-*`/`--inner-*` override is
given), training runs **nested cross-validation** and writes
`raw_predictions.pkl` plus AUC-weight-ensembled per-split predictions.

## `seizure-pred predict`

```bash
seizure-pred predict --config examples/config_prediction.yaml \
  --checkpoint runs/<run_name>/<stamp>
```

`--checkpoint` may be a `.pt` file **or** a run/stamp directory (in which case
all `split_*` folders are predicted; if a split has `inner_split_*`
sub-folders, the inner models are AUC-weight ensembled). Flags: `--split-index`,
`--n-folds`, `--dataloader`, `--mil`, `--strict`, `--threshold`, `--out-dir`,
`--apply-postprocess` (applies the `postprocess` config block).

## `seizure-pred analyze`

```bash
seizure-pred analyze --run-dir runs/<run_name>/<stamp>
```

For each `split_*` folder it writes `analysis/` reports + ROC/PR/confusion/
history plots. It then runs an aggregated **MA × threshold** sweep with a
Pareto frontier. If `raw_predictions.pkl` exists (nested CV), it additionally
runs the **calibration × MA × threshold** sweep. Nested-CV splits also receive
a `preictal_prob_split_X.png` plot comparing each inner model with the outer
ensemble; `--no-plots` disables this artifact as well.

Flags: `--out-dir`, `--threshold`, `--prefer-postprocessed`, `--no-plots`,
`--sampling-period`, `--calibration-methods`, `--ma-windows`, `--thresholds`,
`--percentiles`, `--suppression-duration`.

Example:

```bash
seizure-pred analyze --run-dir runs/nested/20240101_120000 \
  --calibration-methods none percentile beta isotonic temperature \
  --ma-windows 1 3 5 7 10 --thresholds 0.3 0.4 0.5 0.6 0.7 \
  --percentiles 5 10 15 20 --suppression-duration 60
```

## `seizure-pred experiments`

Run a quick hyper-parameter grid from a base config:

```bash
seizure-pred experiments --config examples/config_prediction.yaml \
  --grid '{"optim.lr": [1e-3, 3e-4], "loss.name": ["bce_logits", "focal"]}' \
  --split-index 0 --n-folds 5
```

Grid keys are dot-paths into the config. See [Experiments](experiments.md).

## `seizure-pred benchmark`

Benchmark registered models:

```bash
seizure-pred benchmark --models eegnet eegwavenet mb_dmgc_cwtffnet \
  --batch-sizes 1 32 --n-runs 100 --output-dir benchmark_results
```

Reports params, FLOPs/MACs (`thop`), CPU/GPU latency + throughput, GPU memory.
See [Benchmarking](benchmark.md).
