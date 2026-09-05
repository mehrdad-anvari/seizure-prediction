# AGENTS.md

## Project

`seizure-pred` is a modular Python library for seizure prediction and detection from windowed EEG data.

The main package is:

```text
src/seizure_pred/
├── analysis/        # Evaluation and visualization
├── cli/             # CLI commands
├── core/            # Shared abstractions and configuration
├── data/             # Dataset loading and splitting
├── experiments/      # Experiment orchestration
├── inference/        # Inference, post-processing, calibration
├── legacy/           # Historical/deprecated implementations
├── models/           # Model architectures
├── preprocessing/    # Dataset preprocessing
├── templates/        # Extension templates
├── training/         # Training infrastructure
└── transforms/       # Reusable data/signal transforms
```

Other important directories include `configs/`, `tests/`, `docs/`, `examples/`, `notebooks/`, and `runs/`.

## Source of Truth

Inspect the repository before making assumptions.

- `pyproject.toml` is the authoritative source for package metadata, dependencies, Python requirements, and optional extras.
- Do not rely on installation/dependency instructions in `README.md` without verifying them against `pyproject.toml`.
- `requirements.txt`, when present, may describe a local environment rather than the project's dependency specification.
- Prefer current code and tests over stale documentation.
- Check existing callers, configs, examples, and relevant documentation before changing public behavior.

## Architecture

Keep responsibilities separated:

- `data/` — dataset loading and splitting.
- `preprocessing/` — dataset-specific preparation.
- `transforms/` — reusable transformations.
- `models/` — model architectures.
- `training/` — training infrastructure.
- `inference/` — inference and prediction processing.
- `analysis/` — evaluation and visualization.
- `experiments/` — experiment orchestration.
- `cli/` — argument parsing and application entry points.

Prefer existing abstractions and registries over introducing parallel mechanisms.

The project uses registries and import-time registration for many components. Preserve the existing lazy-import behavior and registration conventions when adding new components.

## Configuration and APIs

Configuration is dataclass-based and validated. Do not silently accept unknown configuration keys or bypass existing validation.

When adding or changing configurable behavior:

- keep defaults explicit;
- check all callers;
- update relevant examples/docs;
- avoid silently changing experiment behavior.

For model changes, preserve existing input/output contracts unless the change intentionally modifies the API.

## Scientific Reproducibility

This is a research/ML repository. Treat preprocessing, dataset splitting, training, evaluation, and inference behavior as scientific methodology.

Do not change methodology merely to improve a metric.

When modifying scientific behavior, consider:

- dataset and split definitions;
- random seeds;
- preprocessing;
- sampling rate and input shape;
- data leakage;
- temporal ordering;
- training configuration;
- evaluation methodology.

Clearly distinguish implementation bugs from intentional methodological differences.

Do not overwrite previous experiment results unless explicitly requested.

Treat source datasets as immutable. Generated data and experiment outputs should remain separate from source data.

## Changes

Keep changes focused and minimal.

Before modifying shared functionality:

1. Search for its usages.
2. Inspect related tests and configuration.
3. Make the smallest correct change.
4. Run relevant tests or a smoke test.

Do not perform unrelated refactoring, formatting, or dependency changes.

Preserve unrelated local changes.

## Legacy Code

`src/seizure_pred/legacy/` contains historical implementations.

Do not use legacy code as the default implementation for new functionality. Modify it only when compatibility, migration, or reproducibility requires it.

## Testing

Use the project's configured test suite rather than assuming a particular environment.

Run the most relevant tests after changes and add regression tests for bugs when practical.

For data-processing and ML changes, perform a small end-to-end or smoke test when appropriate.

Do not consider a change validated merely because the package imports successfully.

## Data and Signal Processing

For temporal or signal data, be explicit about:

- sampling frequency;
- units;
- timestamps;
- window boundaries;
- annotations;
- missing intervals;
- train/validation/test separation.

Avoid silently changing data interpretation or preprocessing semantics.

## Final Check

Before finishing:

- inspect the final diff;
- verify that modified files are intentional;
- run relevant tests;
- report important assumptions or limitations;
- distinguish verified behavior from unverified assumptions;
- do not claim exact reproducibility unless it has actually been established.