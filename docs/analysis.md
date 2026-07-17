# Analysis & calibration

Analysis is split into per-split reports (`seizure_pred.analysis.runner`) and
aggregated clinical sweeps (`seizure_pred.analysis.summary` and
`seizure_pred.analysis.calibration_sweep`).

## Per-split analysis (`analyze_run`)

For each `split_*` directory reads `predictions.jsonl` / `history.jsonl` and
writes, under `<split>/analysis/`:

- `report.json` / `report.txt` — accuracy, precision, recall, F1, confusion.
- `roc.png`, `pr.png`, `confusion.png`, `history.png`.

Metrics are computed without sklearn (rank-based AUC, trapezoid PR).

## Nested-CV preictal probability comparison

When a split contains `inner_split_*/predictions.jsonl`, the `analyze` CLI
also writes `preictal_probability_comparison.png` under that split's
`analysis/` directory. The plot shows each inner-fold probability together
with the outer AUC-weighted ensemble for every preictal event. Samples are
matched by `(event_id, global_epoch_id)` and ordered by
`epoch_index_within_event`. Multiple events are written to separate files.

## Aggregated MA × threshold sweep (`analyze_multi_split_summary`)

Runs across all `split_*` folders. For each `ma_window × threshold` variant it
applies **within-region** moving-average smoothing (no smoothing across
preictal/interictal boundaries), thresholds, and computes:

- AUC (on smoothed probabilities)
- sensitivity (≥1 preictal detected per event)
- FPR/hour (interictal time only)

Aggregated as mean ± std across folds, with a **Pareto frontier**
(sensitivity vs FPR/hour) saved to `pareto_optimal_variants.csv` plus
`analysis_summary.json` and plots.

## Calibration sweep (`analyze_nested_calibration`)

Available for nested-CV runs that produced `raw_predictions.pkl`. Mirrors the
legacy `analyze_results2.py` variant grid:

```
calibration method × moving-average window × threshold
```

For each outer fold it fits a per-inner-fold calibrator on validation
probabilities, transforms the (aligned) test probabilities, then AUC-weight
ensembles the inner folds. Defaults:

| Parameter | Default |
|-----------|---------|
| `--calibration-methods` | `none percentile beta isotonic temperature` |
| `--ma-windows` | `1 3 5 7 10` |
| `--thresholds` | `0.3 0.4 0.5 0.6 0.7` |
| `--percentiles` | `5 10 15 20` |
| `--suppression-duration` | `None` (disabled) |

### Calibration methods

`seizure_pred.inference.calibration.ProbabilityCalibrator(method=...)`:

- **`percentile`** — `expit(a·logit(p)+b)` mapping the
  `(100 - target_preictal_percentile)`-th percentile of preictal validation
  probabilities to 0.5 (reduces false positives). `target_preictal_percentile`.
- **`beta`** — 3-parameter `expit(a + b·log(p) + c·log(1-p))` (scipy).
- **`isotonic`** — non-parametric monotonic regression (scikit-learn).
- **`temperature`** — `expit((logit(p)-b)/T)` with bounds T∈[0.1,10], b∈[-5,5].

`calibrate_ensemble(test_probs_stack, val_probs_list, val_labels_list, val_aucs,
method=...)` fits per-fold and AUC-weight ensembles.

### Outputs (under `<run>/analysis/`)

`variant_summary.csv`, `pareto_optimal_variants.csv`,
`best_variants_{auc,sensitivity,f1,fpr_per_hour}.csv`,
`calibration_comparison.csv`, `ma_window_comparison.csv`,
`threshold_comparison.csv`, `calibration_summary.json`, and plots under
`analysis/plots/` (`calibration_pareto_frontier.png`,
`calibration_comparison.png`).

## Clinical metrics

`seizure_pred.analysis.metrics.clinical_metrics(y_true, y_pred,
sampling_period=5.0, suppression_duration=None)`:

- **sensitivity** — 1.0 if ≥1 preictal window detected per event.
- **fpr_per_hour** — false alarms per hour over interictal time only.
- **fpr_per_hour_suppressed** — FPR/hour after a suppression window: once an
  alarm fires, the next `suppression_duration` windows are not counted as new
  false alarms (the legacy "FPR_sup" metric).

`moving_average_segmented(probs, y, window)` smooths within contiguous
equal-label regions only (no leakage across label boundaries).
