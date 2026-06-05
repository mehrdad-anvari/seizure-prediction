from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, Optional

import math
import torch


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def _to_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return int(x.item())


def _meta_to_jsonable(meta: Any) -> Any:
    # Keep meta lightweight; analysis doesn't require a strict schema.
    # Lists of dicts are allowed (MIL).
    return meta


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: Iterable,
    *,
    device: str | torch.device = "cpu",
    is_mil: bool = False,
    threshold: float = 0.5,
    postprocess: Optional[object] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield standardized prediction rows.

    Batch contract:
      - Instance: (x, y, meta) with x: (B,C,T) and y: (B,)
      - MIL: (x, y, meta) with x: (B,bag,C,T) and y: (B,)

    Output row schema (see docs):
      y_true, logit, prob, y_pred, optional y_pred_post, meta
    """
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    model.eval()
    model.to(dev)

    # Collect postprocess inputs if requested.
    # Postprocess is expected to have either:
    #  - __call__(labels: list[int]) -> list[int]
    #  - or apply(labels: list[int]) -> list[int]
    pp = postprocess

    pending_rows: list[Dict[str, Any]] = []

    def _apply_postprocess(rows: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
        if pp is None or not rows:
            return rows

        labels = [int(r["y_pred"]) for r in rows]
        if hasattr(pp, "apply"):
            labels_pp = pp.apply(labels)
        else:
            labels_pp = pp(labels)

        for r, lp in zip(rows, labels_pp):
            r["y_pred_post"] = int(lp)
        return rows

    for batch in loader:
        # Expect (x,y,meta) but allow dict-like as long as it has these keys.
        if isinstance(batch, dict):
            x = batch["x"]
            y = batch["y"]
            meta = batch.get("meta")
        else:
            x, y, meta = batch

        x = x.to(dev)
        y = y.to(dev)

        logits = model(x)

        # Some models may return an object with `.logits`
        if hasattr(logits, "logits"):
            logits = logits.logits

        # Ensure shape: (B,)
        if logits.dim() > 1:
            logits = logits.view(logits.size(0), -1).squeeze(-1)

        probs = _sigmoid(logits)
        y_pred = (probs >= threshold).to(torch.int64)

        # Emit rows
        bsz = int(y.shape[0])
        for i in range(bsz):
            row = {
                "y_true": _to_int(y[i]),
                "logit": float(logits[i].detach().cpu().item()),
                "prob": float(probs[i].detach().cpu().item()),
                "y_pred": int(y_pred[i].detach().cpu().item()),
                "meta": _meta_to_jsonable(meta[i] if isinstance(meta, (list, tuple)) else meta),
            }
            pending_rows.append(row)

        # If a postprocessor is present, we apply it in streaming chunks.
        # For window smoothing / hysteresis, it usually needs sequential order;
        # user should ensure loader iteration order corresponds to time order.
        if pp is not None and len(pending_rows) >= 2048:
            for r in _apply_postprocess(pending_rows):
                yield r
            pending_rows = []

    # Flush
    if pending_rows:
        for r in _apply_postprocess(pending_rows):
            yield r
