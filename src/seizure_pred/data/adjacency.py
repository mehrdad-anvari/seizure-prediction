from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def euclidean_dist(positions: np.ndarray) -> np.ndarray:
    """Compute Euclidean distance matrix for (N,2) or (N,3) channel positions."""
    positions = np.asarray(positions, dtype=float)
    diff = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(diff, axis=2)


def inverse_mean_threshold_adjacency(dist_matrix: np.ndarray) -> np.ndarray:
    """Inverse Mean Threshold adjacency (IMT).

    If distance(i,j) is below mean distance from i to others, connect with weight 1/d.
    Symmetrized at the end.
    """
    dist_matrix = np.asarray(dist_matrix, dtype=float)
    n = dist_matrix.shape[0]
    adj = np.zeros((n, n), dtype=float)

    for i in range(n):
        u = dist_matrix[i]
        mean = np.mean(np.delete(u, i))
        for j in range(n):
            if i != j and u[j] < mean:
                adj[i, j] = 1.0 / (u[j] + 1e-6)
        # self-loop weight
        below = u[(u < mean) & (u > 0)]
        if below.size:
            adj[i, i] = 1.0 / float(np.mean(below))
        else:
            adj[i, i] = 1.0

    return np.maximum(adj, adj.T)


def positions_from_standard_1020(channel_names: Iterable[str]) -> np.ndarray:
    """Get approximate channel positions from MNE standard_1020 montage.

    Requires `mne`. If not installed, raises ImportError.
    """
    import mne  # optional dependency

    montage = mne.channels.make_standard_montage("standard_1020")
    pos_dict = montage.get_positions()["ch_pos"]
    pos = []
    for ch in channel_names:
        if ch not in pos_dict:
            raise KeyError(f"Channel '{ch}' not found in standard_1020 montage.")
        p = pos_dict[ch]
        pos.append([p[0], p[1], p[2]])
    return np.asarray(pos, dtype=float)
