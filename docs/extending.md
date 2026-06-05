## Extending

This project is built around **registries** (a lightweight plugin system). To add a new component, implement it and register it under a name:

- Dataset: `DATASETS.register("name")`
- Dataloader: `DATALOADERS.register("name")`
- Model: `MODELS.register("name")`
- Loss: `LOSSES.register("name")`
- Optimizer: `OPTIMIZERS.register("name")`
- Scheduler: `SCHEDULERS.register("name")`
- Evaluator: `EVALUATORS.register("name")`
- Callback: `CALLBACKS.register("name")`
- Postprocess: `POSTPROCESSORS.register("name")`

For a walkthrough, see `docs/PLUGIN_GUIDE.md`.
For starter stubs, see `src/seizure_pred/templates/`.
