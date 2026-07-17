# Trainer contract

The library ships two trainers with the **same interface**:

- `seizure_pred.training.engine.trainer.Trainer` (instance-level)
- `seizure_pred.training.engine.trainer_mil.TrainerMIL` (bag-level; alias
  `MILTrainer`)

## Constructor (keyword-only)

```python
Trainer(
    *,
    model: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[object],
    cfg: TrainConfig,
    run_dir: str,
    artifact_writer: Optional[ArtifactWriter] = None,
    callbacks: Optional[list] = None,
    device: Optional[str] = None,
)
```

## Batch contract

- **Trainer**: `(x, y, meta)` with `x: (B, C, T)`, `y: (B,)` 0/1.
- **TrainerMIL**: `(x, y, meta)` with `x: (B, bag, C, T)`, `y: (B,)` bag labels.

## `fit(*, train_loader, val_loader, write_best_predictions=True) -> str`

Runs `cfg.epochs` epochs. Per epoch:

1. `_train_one_epoch` — forward, loss, backward, grad-clip (`grad_clip_norm`),
   optimizer step, optional step-mode scheduler step. `PreictalWeightedLoss`
   automatically receives temporal weights extracted from `meta`.
2. `evaluate` — forward, loss, `binary_classification_metrics` (acc, precision,
   recall, **f1, auc, ap**, confusion), collects logits/targets/meta.
3. History row appended (`val_auc`, `val_ap`, …).
4. Epoch-mode scheduler step (`ReduceLROnPlateau` falls back to `step(val_loss)`).
5. **Best checkpoint** by the configured `monitor` metric
   (`monitor`/`monitor_mode`); default `val_loss`/`min`, use `auc`/`max` for
   "best by validation AUC". On improvement: save `checkpoints/best.pt`, write
   `metrics.json`, and (optionally) `predictions.jsonl`.

Callbacks fire `on_train_start` / `on_epoch_start` / `on_batch_end` /
`on_val_start` / `on_val_batch_end` / `on_val_end` / `on_epoch_end` /
`on_train_end`. A callback may set `state["stop"] = True` for early stopping.

## `evaluate(val_loader, state=None) -> dict`

Returns `{loss, acc, precision, recall, f1, auc, ap, tp, fp, tn, fn,
val_logits, val_targets, val_meta}`. `_to_binary_logits` normalises model
outputs of shape `(B,)`, `(B,1)`, or `(B,2)` to `(B,)`.

## Checkpoint format

`checkpoints/best.pt` (and a `last.pt` fallback) is a dict:
`{schema_version, saved_at, epoch, model_state_dict, optimizer_state_dict?,
scheduler_state_dict?, metrics}`. Restore with
`seizure_pred.training.engine.checkpoint.restore_checkpoint(path, model=...)`.

## Custom trainers

Implement the same `fit(*, train_loader, val_loader, ...) -> ckpt_path` shape
and return a best-checkpoint path; the CLI/analysis tools key off the standard
run artifacts.
