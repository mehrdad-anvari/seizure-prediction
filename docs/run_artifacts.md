# Run Artifacts

Each training and prediction run writes a standardized **run directory** (printed at the end of `seizure-pred train`). The layout is stable so analysis tools and scripts can reliably process outputs across versions.

## Run Directory Layout

A typical run directory contains:

- **`schema.json`**: Schema version and minimal metadata (written by `ArtifactWriter.write_schema()`).
- **`config.json`**: Full resolved config snapshot used for the run.
- **`history.jsonl`**: JSON lines containing epoch-level/step-level logs (e.g., loss, learning rate, and validation metrics per epoch).
- **`metrics.json`**: Final evaluation metrics (e.g., final loss, best monitored metrics).
- **`predictions.jsonl`**: Optional per-window prediction rows (saved by training evaluators or the predict command). Each row is a JSON object containing at least:
  - `y_true`: integer (0 or 1) indicating the true label.
  - `y_score`: float indicating the predicted probability/score for class 1.
  - plus optional metadata fields (e.g., `subject`, `session`, `start_time`).
- **`checkpoints/`**: Directory containing model weights:
  - `best.pt`: Best model (by monitored validation metric).
  - `last.pt`: Last epoch model.

## Analysis Artifacts

Analysis tools typically write outputs under an `analysis/` sub-directory within the run folder:

- **`analysis/report.json`**: Overall evaluation summary metrics and classification reports.
- **`analysis/*.png`**: Plot files (e.g., ROC curve, PR curve) if plotting is enabled.

## Implementation Details

All run artifacts are managed via the writer class: [ArtifactWriter](src/seizure_pred/training/engine/artifacts.py).
