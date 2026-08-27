# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]
### Added
- Two attention models ported into the zoo and registered under `MODELS`:
  `darnet` (DARNet, Yan et al., NeurIPS 2024) and `mhanet` (MHANet, Li et al.,
  IJCAI 2025). Both take raw `(B, C, T)` windows; `mhanet` needs `einops` and a
  `chunk_size` equal to the window length.
- Five models reconstructed from papers in `papers/` that publish no reference
  code — `fapex` (NeurIPS 2025), `md_rescapsnet` (BSPC 2026), `seizurenet_kan`
  (JESTCH 2026), `seresnet3d` (iScience 2025) and `sbtm` (Sci. Reports 2026).
  Their STFT / PLV-graph / feature front-ends run inside the model, so they
  consume the same `(B, C, T)` windows as the rest of the zoo. Every choice not
  stated by its paper is listed in the module docstring; `sbtm` omits the
  paper's metaheuristic optimiser, which is a training procedure rather than
  part of the network.
- `capsule_margin` loss (Sabour-style margin loss on capsule norms) for
  `md_rescapsnet`, and `compute_csp_filters` to fit that model's CSP projection
  on a training split.
- Validation AUC and average precision are now computed and logged every epoch
  (`val_auc`, `val_ap` in `history.jsonl` and `metrics.json`); previously AUC
  was silently absent (always 0.5) which broke AUC-weighted ensembling and
  nested-CV best-fold selection.
- Configurable best-checkpoint selection via `monitor` / `monitor_mode`
  (`val_loss`/`min` default; `auc`/`max` matches the legacy best-by-val-AUC).
- `PostprocessConfig` field on `TrainConfig` so inference post-processing is
  fully configurable and validated (used by `predict --apply-postprocess`).
- Probability calibration module
  (`seizure_pred.inference.calibration`): `percentile`, `beta`, `isotonic`,
  `temperature` + `calibrate_ensemble` (AUC-weighted, fit-on-validation).
- Calibration-aware analysis sweep
  (`seizure_pred.analysis.calibration_sweep.analyze_nested_calibration`)
  consuming nested-CV `raw_predictions.pkl`, with a
  calibration × MA × threshold variant grid, best-variant CSVs, and a Pareto
  frontier. Wired into `seizure-pred analyze` with new flags
  (`--calibration-methods`, `--ma-windows`, `--thresholds`, `--percentiles`,
  `--suppression-duration`).
- Suppression-based FPR/hour metric (`fpr_per_hour_suppressed`) in
  `clinical_metrics`.
- `nearest` interictal strategy for leave-one-preictal-out CV.
- Model benchmarking utility (`seizure_pred.experiments.benchmark`) and
  `seizure-pred benchmark` CLI (params, FLOPs, CPU/GPU latency, GPU memory).
- Optional XAI module (`seizure_pred.inference.xai`) using Captum
  IntegratedGradients.
- Comprehensive docs (models, transforms, CV, analysis, benchmark, XAI,
  predictions schema, trainer contract, plugin guide) and a rewritten README.

### Changed
- `validate_dict` now resolves `from __future__ import annotations` string
  annotations via `get_type_hints`, so nested dataclass and type validation
  actually works (previously silently skipped).
- The leave-one-preictal splitter is robust to datasets where positive and
  negative samples share a `group_id` (it now intersects with `pos_mask`).
- Rewrote `experiments.grid.run_grid` to use the real Trainer/registry/
  build_loader APIs (it was previously non-functional).

### Fixed
- `experiments/grid.py` constructor/registry/loader/fit call mismatches.
- Example configs used the wrong field name `n_fold` (now `n_folds`).
- Test `conftest` no longer hard-imports `mne`; tests skip gracefully when the
  optional `.[eeg]` extra is missing.

## [0.2.0] - 2025-12-20
### Added
- Converted repository into a pip-installable library (src/ layout) with a stable public API surface.
- Registry/factory plugin system for datasets, dataloaders, models, losses, optimizers, schedulers, evaluators, callbacks, and post-processing.
- Trainer and MIL Trainer with standardized run artifacts (config/history/metrics/predictions) and schema versioning.
- Separate inference API and `seizure-pred predict` CLI.
- `seizure-pred analyze` CLI to generate reports and plots from run artifacts.
- CHB-MIT preprocessing pipeline (BIDS → NPZ) behind optional dependency extra `.[eeg]`.
- CI smoke test plumbing and templates to add new plugins with minimal conflicts.
