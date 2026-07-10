# seizure-pred

Welcome to the documentation for **seizure-pred**, a modular seizure
prediction and detection library for CHB-MIT and related EEG datasets. It is a
clean, plugin-based refactor of the older `CP-DMGC-CWT` research codebase.

## Feature overview

- **Pluggable training**: registries for dataset, dataloader, model, loss,
  optimizer, scheduler, evaluator, callback, postprocessor.
- **Tasks**: seizure *prediction* (preictal vs interictal) and *detection*
  (seizure vs background).
- **Nested cross-validation**: outer × inner loops with leave-one-preictal-out
  and custom stratified K-Fold splitters.
- **Imbalance handling**: undersampling, MIL bags, `pos_weight` auto, focal,
  temporal preictal weighting.
- **Inference**: `predict` / `predict_ensemble` with streaming post-processing
  (threshold, moving-average, hysteresis, compose).
- **Calibration**: percentile / beta / isotonic / temperature (fit-on-val,
  AUC-weighted ensemble).
- **Analysis**: reports, ROC/PR/confusion curves, calibration × MA × threshold
  sweep, Pareto frontier, suppression-based FPR/hour.
- **Benchmarking**: params, FLOPs, CPU/GPU latency, GPU memory.
- **XAI**: Captum IntegratedGradients attributions (optional).
- **22 models** including graph models and the flagship `mb_dmgc_cwtffnet`.

## Quick links

- [Installation Guide](install.md)
- [Command-Line Interface (CLI)](cli.md)
- [Configuration & API Reference](api.md)
- [Model zoo](models.md)
- [Transforms](transforms.md)
- [Cross-validation & Splits](cv.md)
- [Analysis & Calibration](analysis.md)
- [Benchmarking](benchmark.md)
- [Explainability (XAI)](xai.md)
- [Grid experiments](experiments.md)
- [Data Contracts](data_contracts.md)
- [Configuration Validation](config_validation.md)
- [Reproducibility & Determinism](determinism.md)
- [Run artifacts](run_artifacts.md)
- [Predictions schema](predictions_schema.md)
- [Trainer contract](trainer_contract.md)
- [Extending the library](extending.md)
- [Plugin guide](plugin_guide.md)
- [Migration guide](migration.md)
- [Analysis migration guide](analysis_migration.md)
