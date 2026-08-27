from __future__ import annotations

r"""SeizureNet-KAN: graph EEG analysis with Kolmogorov-Arnold networks.

- Paper: Ben Atitallah S, Driss M, Boulila W, Koubaa A. Graph-based EEG analysis
  for seizure prediction enhanced with Kolmogorov-Arnold Networks and
  Self-Supervised Learning. Engineering Science and Technology, an International
  Journal, 73 (2026) 102245.
- Related project: none published.

Reconstruction from the paper text. The pipeline of Fig. 1:

1. **Graph construction** (Eqs. 6-10). Nodes are EEG channels; node attributes are
   five per-channel statistics (mean, variance, skewness, kurtosis,
   peak-to-peak); edges come from the phase-locking value computed on Hilbert
   phases, kept only where ``PLV >= median(PLV) + std(PLV)`` and weighted by the
   PLV itself.
2. **KAGCN encoder** (Eqs. 3-4). A node encoder followed by two graph
   convolutions in which the usual ``W`` + ReLU is replaced by a KAN layer of
   learnable B-splines, each followed by SiLU and dropout, then global mean
   pooling and a KAN readout.
3. **KAN decoder** (Eq. 5), used by the paper's self-supervised attribute-masking
   task to reconstruct masked node features. Provided as :class:`KANDecoder`;
   the SSL pretraining loop itself is a training concern, not part of the model.

Stated by the paper and implemented as such: B-splines of degree 3 with grid size
20, two KAGCN layers, SiLU + dropout, global mean pooling, the five node
statistics, and the median+std PLV threshold.

Inferred, because the paper does not state them:

- Hidden width (64) and the depth of the KAN readout (two layers).
- Batch normalisation on the node encoder output. B-spline activations only act
  on a bounded grid, and raw EEG statistics (a variance next to a skewness)
  span orders of magnitude, so unnormalised inputs would fall outside the spline
  support and reduce every KAN layer to its linear base branch.
- The KAN layer follows the standard formulation ``phi(x) = W_base silu(x) +
  W_spline B(x)``, since the paper cites the reference KAN work without
  restating the parameterisation.

Graphs are built inside the model from ``(B, C, T)`` raw EEG, so no external
graph construction step or ``torch_geometric`` install is needed; with a channel
count in the tens, dense propagation is cheaper than sparse gather/scatter.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def hilbert_phase(x: Tensor) -> Tensor:
    r"""Instantaneous phase via the analytic signal (Eq. 6).

    Builds the analytic signal with the FFT one-sided doubling used by
    ``scipy.signal.hilbert``, so no SciPy dependency is required.

    Args:
        x (torch.Tensor): real signal, ``(..., T)``.

    Returns:
        torch.Tensor: phase in radians, ``(..., T)``.
    """
    n = x.shape[-1]
    spectrum = torch.fft.fft(x, dim=-1)

    weights = x.new_zeros(n)
    if n % 2 == 0:
        weights[0] = 1.0
        weights[n // 2] = 1.0
        weights[1: n // 2] = 2.0
    else:
        weights[0] = 1.0
        weights[1: (n + 1) // 2] = 2.0

    analytic = torch.fft.ifft(spectrum * weights, dim=-1)
    return torch.atan2(analytic.imag, analytic.real + 1e-12)


def plv_adjacency(x: Tensor, *, threshold: bool = True) -> Tensor:
    r"""Phase-locking-value adjacency with the median+std threshold (Eqs. 8-10).

    Args:
        x (torch.Tensor): EEG window, ``(B, C, T)``.
        threshold (bool): apply ``tau = median(PLV) + std(PLV)``; when
            :obj:`False` the raw PLV matrix is returned. (default: :obj:`True`)

    Returns:
        torch.Tensor: weighted adjacency, ``(B, C, C)``, zero diagonal.
    """
    phase = hilbert_phase(x)
    unit = torch.polar(torch.ones_like(phase), phase)  # e^{i phi}
    n = phase.shape[-1]

    # PLV_ij = |mean_t e^{i(phi_i - phi_j)}|
    plv = torch.einsum("bct,bdt->bcd", unit, unit.conj()).abs() / n
    eye = torch.eye(plv.shape[-1], device=x.device, dtype=plv.dtype)
    plv = plv * (1.0 - eye)  # drop self-PLV, which is always 1

    if threshold:
        # Statistics over off-diagonal entries only: the zeroed diagonal would
        # otherwise drag the median down.
        off_diag = plv[:, ~eye.bool()].reshape(plv.shape[0], -1)
        tau = off_diag.median(dim=1).values + off_diag.std(dim=1)
        plv = torch.where(plv >= tau[:, None, None], plv, torch.zeros_like(plv))
    return plv


def statistical_node_features(x: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Per-channel node attributes: mean, variance, skewness, kurtosis, P2P.

    Args:
        x (torch.Tensor): EEG window, ``(B, C, T)``.
        eps (float): guard for the standardised moments. (default: :obj:`1e-8`)

    Returns:
        torch.Tensor: node features, ``(B, C, 5)``.
    """
    mean = x.mean(dim=-1)
    centred = x - mean.unsqueeze(-1)
    var = centred.pow(2).mean(dim=-1)
    std = (var + eps).sqrt()
    skew = (centred.pow(3).mean(dim=-1)) / std.pow(3).clamp_min(eps)
    kurt = (centred.pow(4).mean(dim=-1)) / var.pow(2).clamp_min(eps)
    p2p = x.amax(dim=-1) - x.amin(dim=-1)
    return torch.stack([mean, var, skew, kurt, p2p], dim=-1)


def normalized_adjacency(adj: Tensor) -> Tensor:
    r""":math:`\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}` with self-loops (Eq. 4)."""
    eye = torch.eye(adj.shape[-1], device=adj.device, dtype=adj.dtype)
    adj = adj + eye  # A_tilde = A + I
    deg_inv_sqrt = adj.sum(dim=-1).clamp_min(1e-12).rsqrt()
    return deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)


# --------------------------------------------------------------------------- #
# Kolmogorov-Arnold layers
# --------------------------------------------------------------------------- #
class KANLinear(nn.Module):
    r"""KAN layer with learnable B-spline activations.

    Implements :math:`\phi(x) = W_{base}\,\mathrm{silu}(x) + W_{spline}B(x)`,
    where ``B`` are B-spline bases of degree ``spline_order`` on a uniform grid.
    Each input/output pair therefore owns its own learnable univariate function,
    which is what distinguishes a KAN layer from a linear layer plus activation.

    Args:
        in_features (int): input width ``d``.
        out_features (int): output width ``d'``.
        grid_size (int): number of spline intervals ``G``. (default: :obj:`20`)
        spline_order (int): B-spline degree ``k``. (default: :obj:`3`)
        grid_range (tuple): interval the grid spans. (default: :obj:`(-1.0, 1.0)`)
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 20,
                 spline_order: int = 3, grid_range: Tuple[float, float] = (-1.0, 1.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        step = (grid_range[1] - grid_range[0]) / grid_size
        knots = (
            torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32) * step
            + grid_range[0]
        )
        self.register_buffer("grid", knots.expand(in_features, -1).contiguous())

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, std=0.1 / math.sqrt(in_features))

    def b_splines(self, x: Tensor) -> Tensor:
        """Cox-de Boor recursion; ``(..., in_features)`` -> ``(..., in_features, G + k)``."""
        grid = self.grid  # (in_features, G + 2k + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            left = (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)])
            right = (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:-k])
            bases = left * bases[..., :-1] + right * bases[..., 1:]
        return bases

    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.in_features)
        base = F.linear(F.silu(flat), self.base_weight)
        spline = F.linear(
            self.b_splines(flat).flatten(1),
            self.spline_weight.view(self.out_features, -1),
        )
        return (base + spline).reshape(*shape[:-1], self.out_features)


class KAGCNLayer(nn.Module):
    r"""KAN-enhanced graph convolution: :math:`H^{(l)} = \Phi^{(l)}(\hat{A}H^{(l-1)})` (Eq. 4).

    Args:
        in_features (int): input node width.
        out_features (int): output node width.
        grid_size (int): spline intervals of the KAN layer. (default: :obj:`20`)
        spline_order (int): B-spline degree. (default: :obj:`3`)
    """

    def __init__(self, in_features: int, out_features: int, grid_size: int = 20,
                 spline_order: int = 3):
        super().__init__()
        self.kan = KANLinear(in_features, out_features, grid_size, spline_order)

    def forward(self, h: Tensor, adj_norm: Tensor) -> Tensor:
        """``h``: ``(B, C, F)``, ``adj_norm``: ``(B, C, C)`` -> ``(B, C, F')``."""
        return self.kan(torch.bmm(adj_norm, h))


class KANDecoder(nn.Module):
    r"""Two-layer KAN decoder for masked-attribute reconstruction (Eq. 5).

    :math:`\hat{x} = \Phi(W_4\,\Phi(W_3 z))`. Used by the paper's self-supervised
    generative task; unused on the supervised path.

    Args:
        hidden (int): latent width of ``z``.
        out_features (int): number of node attributes to reconstruct.
        grid_size (int): spline intervals. (default: :obj:`20`)
        spline_order (int): B-spline degree. (default: :obj:`3`)
    """

    def __init__(self, hidden: int, out_features: int, grid_size: int = 20,
                 spline_order: int = 3):
        super().__init__()
        self.layer1 = KANLinear(hidden, hidden, grid_size, spline_order)
        self.layer2 = KANLinear(hidden, out_features, grid_size, spline_order)

    def forward(self, z: Tensor) -> Tensor:
        return self.layer2(self.layer1(z))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class SeizureNetKAN(nn.Module):
    r'''
    SeizureNet-KAN (Ben Atitallah et al., JESTCH 2026): PLV graphs over EEG
    channels encoded by KAN-enhanced graph convolutions.

    Reconstructed from the paper text; see the module docstring for inferred
    choices. Graph construction runs inside ``forward``, so the model consumes
    ordinary ``(B, C, T)`` windows.

    .. code-block:: yaml

        model:
          name: seizurenet_kan
          num_classes: 1
          kwargs:
            hidden: 64
            grid_size: 20
            spline_order: 3

    .. code-block:: python

        model = SeizureNetKAN(num_classes=1)
        logits = model(torch.randn(4, 18, 640))  # (4, 1)

        # Self-supervised attribute masking (paper Sec. 4.3): reconstruct
        # masked node features from the node embeddings.
        node_z = model.node_embeddings(torch.randn(4, 18, 640))
        recon = model.decoder(node_z)

    Args:
        num_classes (int): output logits; use :obj:`1` for a single binary logit. (default: :obj:`2`)
        node_features (int): number of per-channel statistics. (default: :obj:`5`)
        hidden (int): node embedding width. (default: :obj:`64`)
        num_layers (int): number of KAGCN layers. (default: :obj:`2`)
        grid_size (int): B-spline grid size ``G``. (default: :obj:`20`)
        spline_order (int): B-spline degree ``k``. (default: :obj:`3`)
        dropout (float): dropout after each KAGCN layer. (default: :obj:`0.2`)
        threshold_plv (bool): apply the median+std PLV threshold of Eq. 10. (default: :obj:`True`)
        readout_layers (int): KAN layers in the readout head. (default: :obj:`2`)
    '''

    def __init__(self,
                 num_classes: int = 2,
                 node_features: int = 5,
                 hidden: int = 64,
                 num_layers: int = 2,
                 grid_size: int = 20,
                 spline_order: int = 3,
                 dropout: float = 0.2,
                 threshold_plv: bool = True,
                 readout_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if readout_layers < 1:
            raise ValueError(f"readout_layers must be >= 1, got {readout_layers}")

        self.num_classes = num_classes
        self.node_features = node_features
        self.hidden = hidden
        self.threshold_plv = threshold_plv

        # Node encoder + normalisation: keeps node attributes inside the spline grid.
        self.node_encoder = nn.Linear(node_features, hidden)
        self.node_norm = nn.BatchNorm1d(hidden)

        self.layers = nn.ModuleList([
            KAGCNLayer(hidden, hidden, grid_size, spline_order) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

        readout = []
        for _ in range(readout_layers - 1):
            readout.append(KANLinear(hidden, hidden, grid_size, spline_order))
        readout.append(KANLinear(hidden, num_classes, grid_size, spline_order))
        self.readout = nn.ModuleList(readout)

        self.decoder = KANDecoder(hidden, node_features, grid_size, spline_order)

    def node_embeddings(self, x: Tensor, node_mask: Optional[Tensor] = None) -> Tensor:
        """Run graph construction and the encoder; returns ``(B, C, hidden)``.

        Args:
            x (torch.Tensor): EEG window, ``(B, C, T)``.
            node_mask (torch.Tensor, optional): boolean ``(B, C)`` mask; masked
                nodes have their attributes zeroed, which is the paper's
                attribute-masking pretext task.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T) input, got shape {tuple(x.shape)}")

        feats = statistical_node_features(x)  # (B, C, 5)
        if node_mask is not None:
            feats = feats * (~node_mask).unsqueeze(-1).to(feats.dtype)

        adj_norm = normalized_adjacency(plv_adjacency(x, threshold=self.threshold_plv))

        h = self.node_encoder(feats)
        h = self.node_norm(h.transpose(1, 2)).transpose(1, 2)

        for layer in self.layers:
            h = self.dropout(F.silu(layer(h, adj_norm)))
        return h

    def forward(self, x: Tensor) -> Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG window, shape :obj:`(B, C, T)`.

        Returns:
            torch.Tensor: logits, shape :obj:`(B, num_classes)`.
        '''
        h = self.node_embeddings(x)
        z = h.mean(dim=1)  # global mean pooling over channels

        for i, layer in enumerate(self.readout):
            z = layer(z)
            if i != len(self.readout) - 1:
                z = self.dropout(F.silu(z))
        return z


# ---- seizure_pred registry glue ----
from seizure_pred.core.config import ModelConfig
from seizure_pred.training.registries import MODELS


@MODELS.register("seizurenet_kan", help="SeizureNet-KAN: PLV graph + KAN-enhanced GCN (JESTCH 2026).")
def build_seizurenet_kan(cfg: ModelConfig) -> nn.Module:
    kw = dict(getattr(cfg, "kwargs", {}) or {})
    return SeizureNetKAN(
        num_classes=int(getattr(cfg, "num_classes", 2)),
        node_features=int(kw.get("node_features", 5)),
        hidden=int(kw.get("hidden", 64)),
        num_layers=int(kw.get("num_layers", 2)),
        grid_size=int(kw.get("grid_size", 20)),
        spline_order=int(kw.get("spline_order", 3)),
        dropout=float(kw.get("dropout", 0.2)),
        threshold_plv=bool(kw.get("threshold_plv", True)),
        readout_layers=int(kw.get("readout_layers", 2)),
    )
