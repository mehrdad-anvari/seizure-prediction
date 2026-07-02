# End-to-end: detection (MIL)

This walkthrough trains a MIL model and runs prediction + analysis.

## Train MIL

```bash
seizure-pred train --config examples/config_mil_detection.yaml --mil --dataloader mil
```

Training prints:

- `run_dir=runs/<run_name>/<stamp>`

## Predict MIL

```bash
seizure-pred predict --config examples/config_mil_detection.yaml \
  --checkpoint <RUN_DIR> \
  --mil --dataloader mil
```

## Analyze

```bash
seizure-pred analyze --run-dir <RUN_DIR> --no-plots
```
