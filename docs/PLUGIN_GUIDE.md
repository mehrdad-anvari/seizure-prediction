## Where to put new plugins (conflict-minimizing convention)

To reduce merge conflicts, new components should be added in dedicated folders and registered locally:

- Dataset plugins: `src/seizure_pred/training/datasets/<name>.py`
- Dataloader plugins: `src/seizure_pred/training/dataloaders/<name>.py`
- Model plugins: `src/seizure_pred/models/<name>.py`  (registration inside file)
- Loss/opt/scheduler: `src/seizure_pred/training/components/<type>.py`
- Evaluators: `src/seizure_pred/training/evaluators/<name>.py`
- Callbacks: `src/seizure_pred/training/callbacks/<name>.py`
- Postprocessors: `src/seizure_pred/training/postprocess/<name>.py`

### Registration rule
Each plugin registers itself using the appropriate registry, for example:

```python
from seizure_pred.training.registries import MODELS

@MODELS.register("my_model", help="Short description")
def build_my_model(cfg: ModelConfig):
    ...
