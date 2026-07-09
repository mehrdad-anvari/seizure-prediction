# Migration guide (old scripts → new library)

This document maps typical usage in the original repository to the refactored library.

## Old: `train.py` / `train_mil2.py`

These are now **legacy wrappers** under:

- `seizure_pred.legacy.train`
- `seizure_pred.legacy.train_mil2`

Recommended: switch to the CLI or API.

### New: training via CLI
```bash
seizure-pred train --config examples/config_prediction.yaml
```

### New: training via API
Use `Trainer` + registries to build components:
- `seizure_pred.training.engine.trainer.Trainer`
- `seizure_pred.training.engine.trainer_mil.TrainerMIL`
- `seizure_pred.training.engine.build.build_components`

---

## Old: preprocessing scripts in `dataset/`

### New: preprocessing module + CLI
Module:
- `seizure_pred.preprocessing.process_chbmit_bids_dataset`

Note: preprocessing needs optional deps: `pip install seizure-pred[eeg]`.

CLI:
```bash
seizure-pred preprocess-chbmit --dataset-dir /path/to/BIDS_CHBMIT \
  --subject 1,2,3 \
  --segment-sec 5 \
  --preictal-minutes 15
```

---

## Old: prediction inside `train.py`

### New: dedicated inference API + CLI
API:
- `seizure_pred.inference.predictor.predict`

CLI:
```bash
seizure-pred predict --config ... --checkpoint ... --out-dir ...
```

Post-processing is configurable:
```yaml
postprocess:
  name: hysteresis
  kwargs: { threshold_on: 0.7, threshold_off: 0.3, min_on: 3 }
```

---

## Old: `analyze_results3.py`

### New: `seizure_pred.analysis` + `seizure-pred analyze`
- `seizure_pred.analysis.runner.analyze_run(run_dir, out_dir=...)`
- `seizure-pred analyze --run-dir runs/<run>/split_0`

If you have custom plots in `analyze_results3.py`, move them into:
- `src/seizure_pred/analysis/plots.py` (plot functions)
- `src/seizure_pred/analysis/metrics.py` (metric helpers)
and call them from `analysis/runner.py`.

---

## Old: Nested CV in `train.py`

In the old repository, nested cross-validation (outer and inner loops) was implemented inside `train.py`, and prediction was run on the held-out splits.

### New: Native Nested CV Support

Nested cross-validation is natively supported by the new library when the `cv` configuration block is defined in your config file, or overridden via the CLI:

```bash
seizure-pred train --config config.json \
  --outer-method LOO --inner-method KFold --inner-n-fold 5 --inner-mode per_event_strata
```

During nested CV, models are saved under `split_{outer_idx}/inner_split_{inner_idx}/` and ensembled test predictions (weighted by inner validation AUCs) are automatically generated at `split_{outer_idx}/predictions.jsonl` along with `raw_predictions.pkl` for legacy analysis scripts.

---

## Component mapping quick table

| Old concept | New concept |
|---|---|
| dataset classes in `dataset.py` | `seizure_pred.data.*` + `DATASETS` registry |
| custom DataLoader logic | `seizure_pred.training.dataloaders.*` + `DATALOADERS` |
| model creation in scripts | `seizure_pred.models.*` + `MODELS` |
| loss selection in scripts | `seizure_pred.training.components.losses` + `LOSSES` |
| evaluation code in scripts | `seizure_pred.training.evaluators.*` + `EVALUATORS` |
| plotting script | `seizure_pred.analysis` + analyze CLI |
