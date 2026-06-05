# API reference (high-level)

This is a minimal map of the public APIs you will use most often.

## Config
- `seizure_pred.core.config.TrainConfig`
- `seizure_pred.core.io.load_config(path)`

## Registries
- `seizure_pred.training.registries`:
  - `DATASETS`, `DATALOADERS`, `MODELS`, `LOSSES`, `OPTIMIZERS`, `SCHEDULERS`, `EVALUATORS`, `CALLBACKS`, `POSTPROCESSORS`

## Training
- `seizure_pred.training.engine.trainer.Trainer`
- `seizure_pred.training.engine.trainer_mil.TrainerMIL`
- `seizure_pred.training.engine.pipeline`:
  - `build_dataset(cfg.data)`
  - `iter_splits(dataset)`
  - `build_loader(dataset, cfg.data, split=..., shuffle=..., seed=...)`
  - `build_dataloader(cfg_or_data_cfg, dataset, split=..., shuffle=..., seed=...)` (back-compat)

Notes:
- Most users won’t need to call loader builders directly; the trainers/pipeline utilities handle this.
- `build_dataloader` exists mainly for older code paths and accepts either a full `TrainConfig` or a `DataConfig`.

## Inference
- `seizure_pred.inference.predictor.predict`
- `seizure_pred.inference.postprocess` (postprocessors + composition)

## Analysis
- `seizure_pred.analysis.runner.analyze_run(run_dir, out_dir=...)`
- `seizure_pred.analysis.runs.load_predictions(path)`
  - Use `glob('runs/*/*/split_*')` (or your own run_dir list) to discover run directories.
