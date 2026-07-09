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


def _to_binary_logits(raw: torch.Tensor) -> torch.Tensor:
    """Normalize model output to shape (B,) binary logits.

    Supported outputs:
      - (B,)             -> returned as-is
      - (B,1)            -> squeeze dim=1
      - (B,2)            -> convert to single binary logit using (logit1 - logit0)
      - (B,*,...)        -> tries to squeeze/flatten conservatively
    """
    if not isinstance(raw, torch.Tensor):
        raw = torch.as_tensor(raw)

    if raw.ndim == 1:
        return raw

    if raw.ndim == 2:
        if raw.shape[1] == 1:
            return raw[:, 0]
        if raw.shape[1] == 2:
            return raw[:, 1] - raw[:, 0]
        if raw.shape[0] == 1:
            return raw.reshape(-1)
        raise ValueError(f"Expected binary logits with shape (B,), (B,1) or (B,2) but got {tuple(raw.shape)}")

    # Higher-dim: try squeezing singleton dims after batch.
    x = raw
    while x.ndim > 1 and x.shape[1] == 1:
        x = x.squeeze(1)
    if x.ndim == 1:
        return x
    # As a last resort, flatten everything except batch then reduce if 2-class.
    x2 = x.reshape(x.shape[0], -1)
    if x2.shape[1] == 1:
        return x2[:, 0]
    if x2.shape[1] == 2:
        return x2[:, 1] - x2[:, 0]
    raise ValueError(f"Could not coerce logits to binary: got {tuple(raw.shape)}")


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
        logits = _to_binary_logits(logits)

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


@torch.no_grad()
def predict_ensemble(
    models: list[torch.nn.Module],
    weights: list[float] | Any,
    loader: Iterable,
    *,
    device: str | torch.device = "cpu",
    is_mil: bool = False,
    threshold: float = 0.5,
    postprocess: Optional[object] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield ensembled prediction rows from a list of models."""
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    for m in models:
        m.eval()
        m.to(dev)

    weights_t = torch.tensor(weights, dtype=torch.float32, device=dev)
    if weights_t.sum() > 0:
        weights_t = weights_t / weights_t.sum()
    else:
        weights_t = torch.ones_like(weights_t) / len(weights_t)

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
        if isinstance(batch, dict):
            x = batch["x"]
            y = batch["y"]
            meta = batch.get("meta")
        else:
            x, y, meta = batch

        x = x.to(dev)
        y = y.to(dev)

        # Collect predictions from all models
        probs_all = []
        logits_all = []
        for model in models:
            logits = model(x)
            if hasattr(logits, "logits"):
                logits = logits.logits
            logits = _to_binary_logits(logits)
            probs = torch.sigmoid(logits)
            probs_all.append(probs)
            logits_all.append(logits)

        probs_stack = torch.stack(probs_all, dim=0)   # shape (num_models, B)
        logits_stack = torch.stack(logits_all, dim=0)  # shape (num_models, B)

        # Weighted average over models
        probs_ensemble = (probs_stack * weights_t.view(-1, 1)).sum(dim=0)
        logits_ensemble = (logits_stack * weights_t.view(-1, 1)).sum(dim=0)

        y_pred = (probs_ensemble >= threshold).to(torch.int64)

        # Emit rows
        bsz = int(y.shape[0])
        for i in range(bsz):
            row = {
                "y_true": _to_int(y[i]),
                "logit": float(logits_ensemble[i].detach().cpu().item()),
                "prob": float(probs_ensemble[i].detach().cpu().item()),
                "y_pred": int(y_pred[i].detach().cpu().item()),
                "meta": _meta_to_jsonable(meta[i] if isinstance(meta, (list, tuple)) else meta),
            }
            pending_rows.append(row)

        if pp is not None and len(pending_rows) >= 2048:
            for r in _apply_postprocess(pending_rows):
                yield r
            pending_rows = []

    # Flush
    if pending_rows:
        for r in _apply_postprocess(pending_rows):
            yield r
