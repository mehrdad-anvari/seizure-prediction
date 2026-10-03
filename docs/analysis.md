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
also writes `preictal_prob_split_X.png` under that split's `analysis/`
directory. The plot shows each inner-fold probability together
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

### Additional per-variant metrics

`variant_summary.csv` also carries `mean_/std_` columns (existing columns are unchanged):

- `auprc` — average precision of the smoothed probabilities on the outer test set
  (natural prevalence).
- `balanced_f1`, `balanced_precision`, `balanced_recall`, `balanced_specificity` —
  window-level metrics on a **class-balanced** test set, computed deterministically
  by reweighting (no subsampling): each interictal window gets weight `w = n_pos / n_neg`
  and each preictal window weight 1, so both classes have equal total weight. Weighted
  `precision = TP / (TP + w*FP)`, `recall = TP / (TP + FN)`,
  `specificity = TN / (TN + FP)` (unchanged by the weighting), and `F1` from the weighted
  precision/recall. Folds with a single class give NaN (excluded from the mean); a fold
  with no positive predictions has precision 0 (so F1 0).

### Inner-validation threshold (`val`)

The fixed-threshold sweep is evaluated on the test set itself. To avoid that peeking,
each outer split also gets a validation-derived threshold: for each inner fold, the
threshold (grid 0.01..0.99) maximizing window-level F1 on that fold's own validation
predictions (original windows only) is found; the AUC-weighted mean over inner folds is
the outer split's threshold. It is stored as `val_threshold` per outer fold in
`raw_predictions.pkl`; older pickles without it are handled by recomputing it from the
saved inner `val_probs`/`val_labels`. It is derived from uncalibrated validation
probabilities, so it is mainly meaningful for calibration `none`. Results for every
calibration x MA window are written to `variant_summary_val_threshold.csv` (same columns
plus `mean_/std_threshold_used`), kept separate so `config_threshold` in
`variant_summary.csv` stays numeric. If no inner validation predictions are available the
`val` variants are skipped with a logged note.

### Outputs (under `<run>/analysis/`)

`variant_summary.csv`, `variant_summary_val_threshold.csv`, `pareto_optimal_variants.csv`,
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

**Label leak caveat.** That segmentation uses the ground-truth label, which a
deployed system does not have; with shuffled interictal test windows
(`cv.outer_shuffle`) it also averages random, non-adjacent windows. The opt-in
`seizure-pred analyze --ma-segmentation time` (calibration sweep only; default
`label` = unchanged) instead uses `moving_average_by_segments` with segments from
`segment_ids_from_meta`: a new segment starts when `event_id` changes or
`start_time_in_event` does not advance by `0 < dt <= 1.5 * sampling_period`. It reads
no labels. Results go to `<run_dir>/analysis_ma_time/` (unless `--out-dir`), so the
default `analysis/` files are untouched; the per-split summary sweep is skipped in
this mode. Old pickles without `event_id`/`start_time_in_event` in `test_meta`
abort the time mode with a logged message (no silent label fallback).
