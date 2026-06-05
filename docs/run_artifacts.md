## Run artifacts

Training and prediction write a **run directory** (printed at the end of `seizure-pred train`).
The layout is intentionally stable so analysis tools can work across versions.

Typical contents:

- `schema.json` — schema version + minimal metadata
- `config.json` — resolved config snapshot
- `history.jsonl` — epoch-level logs
- `metrics.json` — best/summary metrics
- `checkpoints/` — model weights (e.g. `best.pt`, `last.pt`)
- `predictions.jsonl` — optional per-window prediction rows (saved by training/predict)

Analysis typically writes into `analysis/` under the run directory:

- `analysis/report.json`
- `analysis/*.png` (if plotting is enabled)

Implementation: `seizure_pred.training.engine.artifacts.ArtifactWriter`.
