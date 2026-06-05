from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset


def estimate_pos_weight(
    dataset: Dataset,
    *,
    label_getter: Optional[str] = None,
    max_samples: int = 200000,
) -> torch.Tensor:
    """Estimate pos_weight for BCEWithLogitsLoss as neg/pos.

    Assumes binary labels {0,1}. For performance, this scans at most max_samples.

    Dataset item contracts supported:
      - (x, y, meta)
      - (x, y)
      - dict with key 'y'
    """
    pos = 0
    neg = 0
    n = 0

    for item in dataset:
        if isinstance(item, dict):
            y = item.get("y")
        else:
            y = item[1]

        # y may be tensor/scalar/array; normalize to int
        if torch.is_tensor(y):
            yv = int(y.detach().cpu().view(-1)[0].item())
        else:
            yv = int(y)

        if yv == 1:
            pos += 1
        else:
            neg += 1

        n += 1
        if n >= max_samples:
            break

    # avoid division by zero
    if pos == 0:
        return torch.tensor(1.0)
    return torch.tensor(float(neg) / float(pos))
