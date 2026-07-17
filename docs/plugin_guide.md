# Plugin guide

`seizure-pred` is built around a small, uniform `Registry` (see
`seizure_pred.core.registry`). Each subsystem has its own registry instance in
`seizure_pred.training.registries`:

```
DATASETS, DATALOADERS, MODELS, LOSSES, OPTIMIZERS, SCHEDULERS,
EVALUATORS, CALLBACKS, POSTPROCESSORS
```

`MODELS.create("eegnet", cfg)` calls the registered factory with the given
args. `MODELS.names()` lists keys; `MODELS.get("name")` returns the
`Entry(factory, help)`.

## Register a new component

Use the `@REG.register("name", help="...")` decorator. Registration happens on
import, so import your module (or call a `register_all()`) before creating.

```python
from seizure_pred.training.registries import MODELS, LOSSES, POSTPROCESSORS

@MODELS.register("my_cnn", help="My CNN")
def build_my_cnn(cfg):
    kw = dict(cfg.kwargs or {})
    return MyCNN(in_channels=cfg.in_channels, num_classes=cfg.num_classes, **kw)

@LOSSES.register("my_loss")
def build_my_loss(**kw):
    return MyLoss(**kw)

@POSTPROCESSORS.register("my_pp", help="My postprocessor")
def build_my_pp(threshold: float = 0.5, **kw):
    return MyPostprocessor(threshold=threshold)
```

## Where built-ins register

`seizure_pred.training.register_all()` imports these packages (side-effect
registration):

- `training.datasets` → `DATASETS`
- `training.dataloaders` → `DATALOADERS`
- `training.components` (losses/optimizers/schedulers/mil) → `LOSSES`,
  `OPTIMIZERS`, `SCHEDULERS`
- `training.evaluators` → `EVALUATORS`
- `training.callbacks` → `CALLBACKS`
- `training.postprocess` → `POSTPROCESSORS`
- `models.register_all()` → `MODELS`

## Optional dependencies

Guard optional imports so a missing dep never breaks unrelated features:

```python
from seizure_pred.core.optional_deps import is_torch_geometric_available
if is_torch_geometric_available():
    from . import rgnn  # registers "rgnn"
```

Use `_require_extra` patterns (see `transforms/registry.py`) to raise a clear
`ImportError` *only when the feature is used*.

## Dataclass & validation

New config knobs must be added to the relevant dataclass in
`seizure_pred.core.config` so that `validate_config_dict` accepts them and
`from_dict` populates them. The validator resolves annotations and rejects
unknown keys / wrong types.

## Templates

See `src/seizure_pred/templates/` for copy-paste starter files for a model,
loss, dataset, dataloader, and postprocessor.
