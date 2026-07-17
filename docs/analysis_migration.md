# Analysis Migration Guide (analyze_results3.py → analysis)

The old `analyze_results3.py` script was an ad-hoc implementation. Its core clinical evaluation metrics and postprocessing sweeps have been integrated directly into the refactored [analysis](src/seizure_pred/analysis) package.

This package provides:

- **[runner.py](src/seizure_pred/analysis/runner.py)**: Exposes `analyze_run(run_dir, out_dir=...)`, serving as the primary analysis entry point.
- **[summary.py](src/seizure_pred/analysis/summary.py)**: Exposes `analyze_multi_split_summary(...)` which automatically aggregates predictions across all splits (e.g. from cross-validation), sweeps moving average window sizes and decision thresholds, identifies the Pareto-optimal frontier, and generates visualizations.
- **[runs.py](src/seizure_pred/analysis/runs.py)**: Loader helpers to parse standardized artifacts (e.g. `load_predictions()`).
- **[metrics.py](src/seizure_pred/analysis/metrics.py)**: Reusable metric calculations, including:
  - standard classification metrics (`binary_report`, `roc_curve`, `pr_curve`, `auc_trapz`)
  - segmented moving average smoothing (`moving_average_segmented`)
  - clinical event-level metrics (`clinical_metrics`) including Sensitivity and False Positive Rate (FPR) per hour.
- **[plots.py](src/seizure_pred/analysis/plots.py)**: Matplotlib plotting helpers designed to run in headless (non-GUI) environments.

## Multi-Split / Cross-Validation Analysis

When you train a model using cross-validation (e.g. with the `--n-folds` flag in the CLI), the directory layout contains multiple `split_X` folders. 

Running the CLI command:
```bash
seizure-pred analyze --run-dir runs/<run_name>/<timestamp>
```
will automatically detect the multiple splits, run the single-split analysis for each split, and then run `analyze_multi_split_summary(...)` on the parent directory.

This creates the following outputs under `<run_dir>/analysis/`:
- `analysis_summary.json`: Full metrics (Mean AUC, Mean Sensitivity, Mean FPR/hour) for all swept moving average window sizes and thresholds.
- `pareto_optimal_variants.csv`: The subset of configurations that form the Pareto frontier (maximizing Sensitivity and minimizing FPR per hour).
- `plots/`:
  - `pareto_frontier.png`: Scatter-line plot showing the trade-off frontier.
  - `threshold_analysis_ma<W>.png`: Plots of AUC, Sensitivity, and FPR/hour vs threshold for each moving average window size.
  - `metrics_vs_window.png`: Line plot comparing metrics across different smoothing window sizes.

For nested cross-validation runs, analysis also compares each outer ensemble
against all of its inner-fold models. For every held-out preictal event it
plots probability over time, aligning samples by `(event_id, global_epoch_id)`
and ordering them by `epoch_index_within_event`. A split containing one event writes:

- `<run_dir>/split_X/analysis/preictal_probability_comparison.png`

If a split contains multiple preictal events, the event ID is appended to each
filename so that probabilities from separate events are never connected.

You can also specify a custom sampling period for the FPR/hour calculation (default is 5.0 seconds per sample window):
```bash
seizure-pred analyze --run-dir runs/<run_name>/<timestamp> --sampling-period 10.0
```

## Python API Usage

You can run the clinical multi-split summary sweeps directly in Python:

```python
from seizure_pred.analysis.summary import analyze_multi_split_summary

summary = analyze_multi_split_summary(
    run_dir="runs/my_experiment/20260710_134512",
    sampling_period=5.0,
    ma_windows=[1, 3, 5, 7, 10],
    thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    make_plots=True,
)
```
