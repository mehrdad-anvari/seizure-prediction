## Trainer wiring contract

All training entrypoints (CLI/API) should instantiate Trainer/TrainerMIL with:

- `cfg: TrainConfig`
- `run_dir: str` (where artifacts/checkpoints go)
- `evaluator: object` created from `EVALUATORS` registry
- `callbacks: list[object]` created from `CALLBACKS` registry
- `artifact_writer: ArtifactWriter` (writes config/history/metrics/predictions/schema)

Trainer should:
- write `history.jsonl` each epoch
- write `metrics.json` on best validation
- write `predictions.jsonl` when configured (or when `save_predictions=True`)
- include schema_version fields for forward compatibility
