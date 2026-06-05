from __future__ import annotations

import numpy as np


def scale_to_uint16(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale a (N, C, T) float array to uint16 per-sample.

    Returns
    -------
    X_uint16:
        uint16 array with same shape as X.
    scales:
        float32 array of shape (N, 2) holding (min, max) per sample.
    """
    if X.ndim != 3:
        raise ValueError(f"Expected (N,C,T) array, got shape {X.shape}")

    mins = X.reshape(X.shape[0], -1).min(axis=1).astype(np.float32)
    maxs = X.reshape(X.shape[0], -1).max(axis=1).astype(np.float32)
    scales = np.stack([mins, maxs], axis=1)

    # Avoid divide by zero: if max==min => all zeros
    span = (maxs - mins)
    span_safe = np.where(span == 0, 1.0, span)

    Xn = (X - mins[:, None, None]) / span_safe[:, None, None]
    X_uint16 = np.clip(np.round(Xn * 65535.0), 0, 65535).astype(np.uint16)

    # For flat signals force zeros (purely cosmetic; reconstruction uses min/max)
    flat_mask = span == 0
    if np.any(flat_mask):
        X_uint16[flat_mask] = 0

    return X_uint16, scales


def invert_uint16_scaling(X_uint16: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Reconstruct float32 EEG from uint16 + per-sample scales.

    Parameters
    ----------
    X_uint16:
        uint16 array of shape (N,C,T)
    scales:
        float array (N,2) with (min,max)
    """
    if X_uint16.ndim != 3:
        raise ValueError(f"Expected (N,C,T) array, got shape {X_uint16.shape}")
    if scales.ndim != 2 or scales.shape[1] != 2:
        raise ValueError(f"Expected scales shape (N,2), got {scales.shape}")
    if scales.shape[0] != X_uint16.shape[0]:
        raise ValueError("scales and X_uint16 must have the same first dimension")

    mins = scales[:, 0].astype(np.float32)
    maxs = scales[:, 1].astype(np.float32)
    span = (maxs - mins)

    Xf = X_uint16.astype(np.float32) / 65535.0
    X_rec = Xf * span[:, None, None] + mins[:, None, None]

    # For flat signals, fill with min
    flat_mask = span == 0
    if np.any(flat_mask):
        X_rec[flat_mask] = mins[flat_mask][:, None, None]

    return X_rec
