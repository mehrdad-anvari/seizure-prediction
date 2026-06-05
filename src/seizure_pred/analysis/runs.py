from __future__ import annotations

import json
from typing import Optional, Tuple

import numpy as np


def load_predictions(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load standardized predictions.jsonl.

    Returns:
      y_true: (N,)
      prob: (N,)
      y_pred: (N,)
      y_pred_post: (N,) or None
    """
    y_true = []
    prob = []
    y_pred = []
    y_pred_post = []

    has_post = False

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            y_true.append(int(row["y_true"]))
            prob.append(float(row.get("prob", 0.0)))
            y_pred.append(int(row.get("y_pred", 0)))

            if "y_pred_post" in row:
                has_post = True
                y_pred_post.append(int(row["y_pred_post"]))

    y_true_a = np.asarray(y_true, dtype=np.int64)
    prob_a = np.asarray(prob, dtype=np.float64)
    y_pred_a = np.asarray(y_pred, dtype=np.int64)

    if has_post:
        y_pred_post_a = np.asarray(y_pred_post, dtype=np.int64)
    else:
        y_pred_post_a = None

    return y_true_a, prob_a, y_pred_a, y_pred_post_a
