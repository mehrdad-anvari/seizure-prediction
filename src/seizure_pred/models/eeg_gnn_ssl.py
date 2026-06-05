from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from seizure_pred.core.config import ModelConfig
from seizure_pred.data.adjacency import euclidean_dist, inverse_mean_threshold_adjacency, positions_from_standard_1020
from seizure_pred.training.registries import MODELS


# ---- Graph / Laplacian helpers (optional SciPy) ----

def _as_sparse_coo(a: np.ndarray):
    import scipy.sparse as sp  # optional dependency
    return sp.coo_matrix(a)


def _scaled_laplacian(adj: np.ndarray, lambda_max: Optional[float] = None) -> np.ndarray:
    """Scaled Laplacian as used in DCRNN / diffusion convolution.

    Requires scipy.
    """
    import scipy.sparse as sp  # optional dependency
    from scipy.sparse import linalg  # optional dependency

    adj = _as_sparse_coo(adj)
    n = adj.shape[0]
    d = np.array(adj.sum(1)).flatten()
    d_inv_sqrt = np.power(d, -0.5, where=(d > 0))
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(n) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    if lambda_max is None:
        try:
            lambda_max, _ = linalg.eigsh(normalized_laplacian, 1, which='LM')
            lambda_max = float(lambda_max[0])
        except Exception:
            lambda_max = 2.0
    L = (2.0 / lambda_max) * normalized_laplacian - sp.eye(n)
    return L.astype(np.float32).toarray()


def _build_supports(adj: np.ndarray) -> torch.Tensor:
    """Return supports tensor (K, N, N). For now K=1 scaled laplacian."""
    L = _scaled_laplacian(adj)
    return torch.from_numpy(L).unsqueeze(0)  # (1, N, N)


# ---- Model components (adapted from eeg-gnn-ssl DCRNN classification) ----

class DiffusionGraphConv(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_nodes: int, num_supports: int = 1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.num_supports = num_supports
        self.theta = nn.Parameter(torch.empty((num_supports, in_dim, out_dim)))
        nn.init.xavier_uniform_(self.theta)

    def forward(self, x: torch.Tensor, supports: torch.Tensor) -> torch.Tensor:
        # x: (B, N, in_dim)
        # supports: (K, N, N)
        K, N, _ = supports.shape
        assert K == self.num_supports
        out = 0.0
        for k in range(K):
            s = supports[k]  # (N,N)
            xk = torch.einsum("nm,bmi->bni", s, x)  # (B,N,in_dim)
            out = out + torch.einsum("bni,io->bno", xk, self.theta[k])
        return out


class DCGRUCell(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_nodes: int, num_supports: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_supports = num_supports

        self.gconv_zr = DiffusionGraphConv(input_dim + hidden_dim, 2 * hidden_dim, num_nodes, num_supports)
        self.gconv_h = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, num_nodes, num_supports)

    def forward(self, x: torch.Tensor, h: torch.Tensor, supports: torch.Tensor) -> torch.Tensor:
        # x: (B,N,input_dim), h: (B,N,hidden_dim)
        inp = torch.cat([x, h], dim=-1)
        zr = torch.sigmoid(self.gconv_zr(inp, supports))
        z, r = torch.split(zr, self.hidden_dim, dim=-1)
        inp2 = torch.cat([x, r * h], dim=-1)
        hc = torch.tanh(self.gconv_h(inp2, supports))
        h_new = (1 - z) * h + z * hc
        return h_new


class DCRNNEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_nodes: int, num_layers: int = 1, num_supports: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            DCGRUCell(input_dim if i == 0 else hidden_dim, hidden_dim, num_nodes, num_supports)
            for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, supports: torch.Tensor) -> torch.Tensor:
        # x: (B,T,N,input_dim)
        B, T, N, D = x.shape
        h = [x.new_zeros((B, N, self.cells[0].hidden_dim)) for _ in range(self.num_layers)]
        for t in range(T):
            inp = x[:, t]
            for i, cell in enumerate(self.cells):
                h[i] = cell(inp, h[i], supports)
                inp = h[i]
        return h[-1]  # (B,N,H)


class DCRNNModelClassification(nn.Module):
    """Graph RNN classifier producing a single logit per sample."""

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        seq_len: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_supports: int = 1,
        fc_dim: int = 128,
        dropout: float = 0.2,
        supports: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.encoder = DCRNNEncoder(input_dim, hidden_dim, num_nodes, num_layers=num_layers, num_supports=num_supports)
        self.fc1 = nn.Linear(num_nodes * hidden_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("_supports", supports if supports is not None else torch.eye(num_nodes).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept x as (B,C,T) and interpret C as nodes and T as time with input_dim=1
        if x.dim() == 3:
            B, C, T = x.shape
            assert C == self.num_nodes, f"Expected C=num_nodes={self.num_nodes}, got {C}"
            # reshape to (B,T,N,1)
            x = x.permute(0, 2, 1).unsqueeze(-1)
        elif x.dim() == 4:
            # (B,T,N,D)
            pass
        else:
            raise ValueError(f"Unsupported input shape {tuple(x.shape)}")

        h = self.encoder(x, self._supports)  # (B,N,H)
        h = h.reshape(h.size(0), -1)
        h = F.relu(self.fc1(h))
        h = self.dropout(h)
        logit = self.fc2(h).squeeze(-1)
        return logit


def _build_adjacency_from_cfg(cfg: ModelConfig) -> np.ndarray:
    kw = dict(cfg.kwargs or {})
    # priority: explicit adjacency matrix
    if "adjacency" in kw:
        adj = np.asarray(kw["adjacency"], dtype=float)
        return adj

    # else: compute from positions/channel names
    if "positions" in kw:
        pos = np.asarray(kw["positions"], dtype=float)
    else:
        ch_names = kw.get("channel_names")
        if ch_names is None:
            raise ImportError(
                "To build 'eeg_gnn_ssl' without providing adjacency, you must pass cfg.kwargs['channel_names'] "
                "and have optional dependency 'mne' installed (for standard_1020 montage), or provide cfg.kwargs['positions']. "
                "Alternatively provide cfg.kwargs['adjacency'] directly."
            )
        pos = positions_from_standard_1020(ch_names)

    dist = euclidean_dist(pos)
    return inverse_mean_threshold_adjacency(dist)


@MODELS.register("eeg_gnn_ssl", help="DCRNN-style graph RNN classifier (optional scipy/mne for adjacency build).")
def build_eeg_gnn_ssl(cfg: ModelConfig) -> nn.Module:
    # NOTE: This model can run without scipy/mne if you pass cfg.kwargs['adjacency'].
    kw = dict(cfg.kwargs or {})

    num_nodes = int(kw.get("num_nodes", getattr(cfg, "in_channels", None) or 18))
    seq_len = int(kw.get("seq_len", kw.get("t", 256)))
    hidden_dim = int(kw.get("hidden_dim", 64))
    num_layers = int(kw.get("num_layers", 1))
    fc_dim = int(kw.get("fc_dim", 128))
    dropout = float(kw.get("dropout", 0.2))

    # Build supports
    try:
        adj = _build_adjacency_from_cfg(cfg)
        supports = _build_supports(adj)
    except ModuleNotFoundError as e:
        raise ImportError(
            "Building adjacency/supports for 'eeg_gnn_ssl' requires optional dependency 'scipy' (and 'mne' if using channel_names). "
            "Install extras or provide cfg.kwargs['adjacency'] directly."
        ) from e

    return DCRNNModelClassification(
        num_nodes=num_nodes,
        input_dim=1,
        seq_len=seq_len,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_supports=int(supports.shape[0]),
        fc_dim=fc_dim,
        dropout=dropout,
        supports=supports,
    )
