# Experiments (Grid Runner)

This library includes a minimal grid runner utility to help automate running multiple experiments across hyperparameter grids without copy-pasting configuration files.

## Example CLI Usage

Run a hyperparameter grid search using the `seizure-pred experiments` command:

```bash
seizure-pred experiments \
  --config examples/config_prediction.yaml \
  --grid '{"optim.lr":[0.001,0.0003], "loss.name":["bce_logits","focal"]}' \
  --split-index 0 --n-folds 5
```

Flags: `--config`, `--grid` (JSON dot-path → list), `--split-index`, `--n-folds`,
`--mil` (use the MIL trainer), `--save-root` (override `config.save_dir`).

### Grid Syntax

The `--grid` option takes a JSON-formatted object where keys are **dot-paths** corresponding to fields in the training configuration.

Common fields to sweep:
- `model.name` (e.g., `["simple_cnn", "eegwavenet"]`)
- `optim.lr` (e.g., `[0.001, 0.0003, 0.0001]`)
- `optim.weight_decay` (e.g., `[0.0, 1e-4]`)
- `loss.name` (e.g., `["bce_logits", "focal"]`)
- `sched.name` (e.g., `[null, "cosine"]`)

## Programmatic API

```python
from seizure_pred.experiments.grid import run_grid
run_dirs = run_grid(
    "examples/config_prediction.yaml",
    {"optim.lr": [1e-3, 3e-4], "loss.name": ["bce_logits", "focal"]},
    split_index=0, n_folds=5,
)
```

`run_grid` registers all plugins, validates each combination, builds the dataset
once, and trains one model per grid point on the chosen split. Override
`dataloader_type` inside the base config (not via a flag).

## Experiment Outputs

The experiments command prints a JSON object summarizing the run paths:

```json
{
  "runs": [
    "runs/pred_demo__grid000__optim-lr-0.001_loss-name-bce_logits/20260623_120000/split_0",
    "runs/pred_demo__grid001__optim-lr-0.0003_loss-name-focal/20260623_120000/split_0"
  ]
}
```

All grid points share the same timestamp and each run directory is populated
with standard artifacts (`config.json`, `history.jsonl`, `metrics.json`, and
`checkpoints/best.pt`).
