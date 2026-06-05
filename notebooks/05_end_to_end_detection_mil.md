# End-to-end: detection (MIL)

This walkthrough trains a MIL model and runs prediction + analysis.

## Train MIL

```bash
seizure-pred train --config examples/config_mil_detection.yaml --mil --dataloader mil --split-index 0
```

Training prints:

- `best_checkpoint=.../checkpoints/best.pt`
- `run_dir=runs/<run_name>/<stamp>/split_0`

## Predict MIL

```bash
seizure-pred predict --config examples/config_mil_detection.yaml \
  --checkpoint <PATH_TO_BEST_CHECKPOINT> \
  --mil --dataloader mil --split-index 0 \
  --out-dir <RUN_DIR>/predict_split_0
```

## Analyze

```bash
seizure-pred analyze --run-dir <RUN_DIR>/predict_split_0 --no-plots
```
