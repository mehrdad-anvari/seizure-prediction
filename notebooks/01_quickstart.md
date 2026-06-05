# 01 — Quickstart

This quickstart shows the smallest end-to-end flow using the built-in **synthetic** dataset.

## Install (editable)

```bash
pip install -e ".[train,viz]"
```

## List available plugins

```bash
seizure-pred list
```

## Train (synthetic)

A ready-to-run synthetic config is included:

- `examples/config_synthetic.yaml`

Run a tiny 2-fold training job:

```bash
seizure-pred train --config examples/config_synthetic.yaml --split-index 0 --n-folds 2
```

Training prints `best_checkpoint` and `run_dir`. Keep those.

## Predict + Analyze

```bash
# Use the printed best checkpoint
seizure-pred predict --config examples/config_synthetic.yaml \
  --checkpoint <PATH_TO_BEST_CHECKPOINT> \
  --split-index 0 --n-folds 2 \
  --out-dir <RUN_DIR>/predict_split_0

# Analyze predictions (writes <OUT_DIR>/analysis/report.json and optional plots)
seizure-pred analyze --run-dir <RUN_DIR>/predict_split_0 --no-plots
```
