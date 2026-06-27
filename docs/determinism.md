# Reproducibility & Determinism

This package provides utilities to help ensure reproducible experiments and deterministic runs.

## Seeding Utilities

Use [seed.py](file:///e:/Projects/seizure/library/seizure-prediction/src/seizure_pred/core/seed.py) to seed Python's built-in `random`, `numpy`, and PyTorch:

```python
from seizure_pred.core.seed import seed_everything

# Seeds all random number generators
seed_everything(seed=42)
```

The training CLI automatically calls `seed_everything(...)` once at startup based on the `seed` parameter in the training config.

## Trade-off: Speed vs. Determinism

Enabling strict determinism in PyTorch can reduce training speed. If maximum performance/speed is preferred over exact reproducibility, configure non-deterministic algorithms by modifying the device/runtime configuration.
