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
seizure-pred train --config examples/config_synthetic.yaml --n-folds 2
```

Training prints `best_checkpoint` and `run_dir`. Keep those.

## Predict + Analyze

```bash
# Predict on all splits:
seizure-pred predict --config examples/config_synthetic.yaml \
  --checkpoint <RUN_DIR>

# Analyze predictions for all splits:
seizure-pred analyze --run-dir <RUN_DIR> --no-plots
```
