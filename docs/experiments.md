# Experiments (Grid Runner)

This library includes a minimal grid runner utility to help automate running multiple experiments across hyperparameter grids without copy-pasting configuration files.

## Example CLI Usage

Run a hyperparameter grid search using the `seizure-pred experiments` command:

```bash
seizure-pred experiments \
  --config examples/config_prediction.yaml \
  --grid '{"optim.lr":[0.001,0.0003], "loss.name":["bce_logits","focal"]}' \
  --split-index 0 \
  --dataloader undersample
```

### Grid Syntax

The `--grid` option takes a JSON-formatted object where keys are **dot-paths** corresponding to fields in the training configuration.

Common fields to sweep:
- `model.name` (e.g., `["simple_cnn", "eegwavenet"]`)
- `optim.lr` (e.g., `[0.001, 0.0003, 0.0001]`)
- `optim.weight_decay` (e.g., `[0.0, 1e-4]`)
- `loss.name` (e.g., `["bce_logits", "focal"]`)
- `sched.name` (e.g., `[null, "cosine"]`)

## Experiment Outputs

The experiments command prints a JSON object summarizing the run paths:

```json
{
  "runs": [
    "runs/pred_demo/20260623_120000/split_0",
    "runs/pred_demo/20260623_120512/split_0"
  ]
}
```

Each run directory is populated with standard artifacts (e.g., `config.json`, `history.jsonl`, `metrics.json`, and training checkpoints).
