# Analysis Migration Guide (analyze_results3.py → analysis)

The old `analyze_results3.py` script was an ad-hoc implementation. The refactored [analysis](file:///e:/Projects/seizure/library/seizure-prediction/src/seizure_pred/analysis) package provides modular, reusable components centered around standardized run directory structures and artifacts:

- **[runner.py](file:///e:/Projects/seizure/library/seizure-prediction/src/seizure_pred/analysis/runner.py)**: Exposes `analyze_run(run_dir, out_dir=...)`, serving as the primary analysis entry point.
- **[runs.py](file:///e:/Projects/seizure/library/seizure-prediction/src/seizure_pred/analysis/runs.py)**: Loader helpers to parse standardized artifacts (e.g. `load_predictions()`).
- **[metrics.py](file:///e:/Projects/seizure/library/seizure-prediction/src/seizure_pred/analysis/metrics.py)**: Reusable metric calculations (e.g., ROC/PR metrics, `binary_classification_metrics`).
- **[plots.py](file:///e:/Projects/seizure/library/seizure-prediction/src/seizure_pred/analysis/plots.py)**: Matplotlib plotting helpers designed to run in headless (non-GUI) environments.

## Run Discovery

The new architecture does not use a dedicated `Run` wrapper class. Instead, each run is tracked as a standard filesystem folder path (e.g., `runs/<run_name>/<timestamp>/split_0`).

You can discover existing runs using a glob pattern:

```python
from pathlib import Path

run_dirs = sorted(Path("runs").glob("*/*/split_*"))
```

## Migration Steps

1. **Domain-Specific Metrics**: Move custom domain metrics (like seizure horizon, SOP/SPH calculations) from `analyze_results3.py` into a new package module (e.g., `src/seizure_pred/analysis/seizure_metrics.py`).
2. **General Plotting**: Keep all plotting code general and reusable under `src/seizure_pred/analysis/plots.py`.
3. **Integration**: Wire your domain-specific metrics directly into `analyze_run(...)` or call them in your scripts after calling the standard runner.
