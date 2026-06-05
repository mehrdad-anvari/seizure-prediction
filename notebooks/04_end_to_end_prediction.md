# End-to-end: CHB-MIT prediction (preprocess → train → predict → analyze)

## 1) Preprocess (BIDS → NPZ)
```bash
seizure-pred preprocess-chbmit \
  --dataset-dir data/BIDS_CHB-MIT \
  --subject 1
```

## 2) Train (CLI)
```bash
seizure-pred train --config examples/config_prediction.yaml --dataloader undersample --split-index 0
```

## 3) Predict

> Tip: `seizure-pred train` prints the resolved `best_checkpoint` and `run_dir` at the end.
```bash
seizure-pred predict --config examples/config_prediction.yaml \
  --checkpoint runs/<run_name>/<timestamp>/split_0/checkpoints/best.pt \
  --split-index 0
```

## 4) Analyze

You can run analysis from the CLI:
```bash
seizure-pred analyze --run-dir runs/<run_name>/<timestamp>/split_0
```

Or in Python (read `predictions.jsonl` and compute basic metrics):
```python
from pathlib import Path
from seizure_pred.analysis.runs import load_predictions
from seizure_pred.analysis.metrics import binary_report

run_dir = Path('runs/<run_name>/<timestamp>/split_0')
y_true, prob, y_pred, y_pred_post = load_predictions(run_dir / 'predictions.jsonl')
report = binary_report(y_true, y_pred)
print(report)
```
