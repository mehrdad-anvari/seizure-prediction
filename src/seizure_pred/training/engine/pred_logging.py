from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch


def to_pred_rows(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    meta: Any,
    extra: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    y_true = y_true.detach().cpu().reshape(-1)
    y_score = y_score.detach().cpu().reshape(-1)
    rows: List[Dict[str, Any]] = []
    extra = extra or {}

    # meta is usually list[dict] or list[list[dict]] (MIL)
    if isinstance(meta, list):
        # for MIL, meta[i] might be list[dict]
        for i in range(len(y_true)):
            row = {"y_true": int(y_true[i].item()), "y_score": float(y_score[i].item())}
            row.update(extra)
            row["meta"] = meta[i]
            rows.append(row)
    else:
        # fallback
        for i in range(len(y_true)):
            row = {"y_true": int(y_true[i].item()), "y_score": float(y_score[i].item())}
            row.update(extra)
            rows.append(row)
    return rows
