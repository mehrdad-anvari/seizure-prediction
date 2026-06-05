# Experiments (Grid Runner)

This project includes a minimal grid runner to avoid copy-pasting configs.

## Example

```bash
seizure-pred experiments \
  --config examples/config_prediction.yaml \
  --grid '{"optim.lr":[0.001,0.0003], "loss.name":["bce_logits","focal"]}' \
  --split-index 0 \
  --dataloader undersample
```

### Grid syntax
`--grid` is a JSON object where keys are **dot-paths** inside config.

Common keys:
- `model.name`
- `optim.lr`
- `optim.weight_decay`
- `loss.name`
- `sched.name`

## Output
Prints a JSON object with `runs: [run_dir1, run_dir2, ...]`.
Each run writes artifacts (`config.json`, `history.jsonl`, etc.).
