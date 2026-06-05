# Migrating analyze_results3.py to seizure_pred.analysis

The old `analyze_results3.py` was an ad-hoc script. The new `seizure_pred.analysis` package provides
a small set of reusable building blocks centered around **standardized run artifacts**:

- `analysis/runner.py`: `analyze_run(run_dir, out_dir=...)` (one-stop analysis entrypoint)
- `analysis/runs.py`: helpers to load standardized artifacts (e.g. `load_predictions()`)
- `analysis/metrics.py`: reusable metrics (ROC/PR helpers, `binary_report`, etc.)
- `analysis/plots.py`: plotting helpers (matplotlib; headless-safe)

## Run discovery

There is no `Run` class in the refactored analysis package. Instead, treat a run as a folder such as:
`runs/<run_name>/<timestamp>/split_0`.

To discover runs you can use a filesystem glob, e.g.:

```python
from pathlib import Path

run_dirs = sorted(Path("runs").glob("*/*/split_*"))
```

## Recommended next steps
1. Copy the parts of `analyze_results3.py` that compute *domain-specific* metrics (e.g., seizure horizon, SOP/SPH)
   into a new module like `seizure_pred/analysis/seizure_metrics.py`.
2. Keep plotting general and reusable in `analysis/plots.py`.
3. Wire domain-specific metrics into `analyze_run(...)` (or call them after `analyze_run` in your own scripts).
