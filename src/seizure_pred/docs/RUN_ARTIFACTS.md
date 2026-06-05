# Run artifacts

Each run directory contains standardized outputs that analysis tools and users can rely on.

## Required
- `config.json` : full resolved config used for the run
- `history.jsonl` : JSON lines with step/epoch logs (loss, lr, etc.)
- `metrics.json` : final evaluation metrics (and any extra keys your task writes)

## Optional but recommended
- `predictions.jsonl` : one row per sample/bag with at least:
  - `y_true` : int (0/1)
  - `y_score`: float score/prob for class 1
  - plus optional metadata fields (`subject`, `session`, `start_time`, ...)

## Checkpoints
- `best.pt` : best model (by monitored metric)
- `last.pt` : last epoch model
