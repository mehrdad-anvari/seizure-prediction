## predictions.jsonl row schema (v1)

Each line is a JSON object with:

Required:
- `schema_version`: int (e.g., 1)
- `index`: int (global running index)
- `y_true`: int (0/1)
- `logit`: float (raw model output before sigmoid)
- `prob`: float (sigmoid(logit))
- `y_pred`: int (0/1 using threshold)

Optional:
- `y_pred_post`: int (0/1 after postprocess)
- `meta`: dict (window/session metadata; may include subject/session/window indices, label name, etc.)

MIL:
- `logit`/`prob`/`y_pred` should refer to the **bag-level** score.
- `meta` may be list-like or include bag composition details.

Analysis tools should:
- prefer `y_pred_post` when present and when user requests postprocessed evaluation,
  otherwise use `y_pred`.
