# Predictions schema

Predictions are written as **JSONL** (`predictions.jsonl`), one row per window.

## Row fields

| Field | Type | Meaning |
|-------|------|---------|
| `index` | int | 0-based window index within the file |
| `split` | string | split name (e.g. `"val"`) when written by the trainer |
| `y_true` | int | ground-truth label (0/1) |
| `target` | int | alias of `y_true` (legacy) |
| `logit` | float | model logit (binary) |
| `prob` | float | sigmoid(logit) probability |
| `y_pred` | int | thresholded prediction at the run threshold |
| `y_pred_post` | int | (optional) prediction after post-processing |
| `meta` | object | per-window metadata (event_id, label, timestamps, …) |

Every row also carries `schema_version` and `written_at`.

## Where predictions live

- Trainer: `<run_dir>/predictions.jsonl` (best-epoch validation predictions).
- `predict` CLI: `<out_dir>/split_<k>/predictions.jsonl` (or
  `<split_dir>/predict_split_<k>/predictions.jsonl`).
- Nested CV: per outer split `<stamp>/split_<k>/predictions.jsonl` holds the
  AUC-weight-ensembled test predictions, and `raw_predictions.pkl` holds the
  raw per-inner-fold val/test arrays used by the calibration sweep.

## Consumers

- `seizure_pred.analysis.runs.load_predictions(path)` →
  `(y_true, prob, y_pred, y_pred_post)`.
- `seizure_pred.analysis.runner.analyze_run` reads `predictions.jsonl`.
- `seizure_pred.analysis.summary.analyze_multi_split_summary` reads one file
  per `split_*` folder.
- `seizure_pred.analysis.calibration_sweep.analyze_nested_calibration` reads
  `raw_predictions.pkl` (the richer nested-CV schema).
