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

    Supports both monopolar channel names (e.g. 'Fp1') and bipolar channel names
    separated by a dash (e.g. 'Fp1-F3'), case-insensitively.
    """
    import mne  # optional dependency

    montage = mne.channels.make_standard_montage("standard_1020")
    pos_dict = montage.get_positions()["ch_pos"]
    
    # Normalize keys in pos_dict to upper case for case-insensitive lookup
    pos_dict_upper = {k.upper(): v for k, v in pos_dict.items()}

    pos = []
    for ch in channel_names:
        ch_clean = str(ch).strip()
        if "-" in ch_clean:
            parts = ch_clean.split("-")
            ch1, ch2 = parts[0].strip(), parts[1].strip()
            p1 = pos_dict_upper.get(ch1.upper())
            p2 = pos_dict_upper.get(ch2.upper())
            if p1 is None or p2 is None:
                raise KeyError(f"Bipolar channel parts '{ch1}' or '{ch2}' for '{ch_clean}' not found in standard_1020 montage.")
            p = (p1 + p2) / 2.0
        else:
            p = pos_dict_upper.get(ch_clean.upper())
            if p is None:
                raise KeyError(f"Channel '{ch_clean}' not found in standard_1020 montage.")
        pos.append([p[0], p[1], p[2]])
    return np.asarray(pos, dtype=float)
