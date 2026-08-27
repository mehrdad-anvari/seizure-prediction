from __future__ import annotations

r"""FAPEX: Fractional Amplitude-Phase Expressor for cross-subject seizure prediction.

- Paper: Zheng R, Mao L, Luo T, Wang Y, Han D, Ding J, Yu Y. FAPEX: Fractional
  Amplitude-Phase Expressor for Robust Cross-Subject Seizure Prediction.
  Advances in Neural Information Processing Systems 39 (NeurIPS 2025).
- Related project: none published.

This is a reconstruction from the paper text (no reference implementation exists).
The three components and the equations they implement:

1. ``FrNFO`` -- fractional neural frame operator (Eqs. 3-5). An implicit MLP
   generates a complex window kernel from Hermite polynomials times complex
   sinusoids; a learnable per-feature fractional order theta in (0, pi) drives a
   fractional-Fourier-domain product, whose magnitude and angle become the
   amplitude and phase streams.
2. ``APCE`` -- amplitude-phase cross-encoding (Eqs. 6-12). Two bidirectional
   selective state-space blocks, where each stream supplies the state-space
   parameters (B, C) for the other, so phase conditions amplitude and vice versa.
3. ``SpatialCorrelationAggregation`` -- SCA (Eq. 13). Linear attention across
   electrodes, gated by a depthwise 3x3 convolution over the (channel, patch)
   plane, modelling inter-electrode dependencies.

Because the paper's appendix (implementation details, App. H) is not part of the
published main PDF, the following are **inferred** and exposed as constructor
arguments rather than taken from the authors:

- ``d_model``, ``d_state``, ``d_inner``, ``depth``, ``patch_size``, ``hermite_order``
  and ``num_sinusoids``. The interpretability figure (Fig. 2) reports FrNFO filter
  responses for "Layer 1 ... Layer 12", so the authors' backbone is deeper than
  the default ``depth=4`` used here to stay affordable on 640-sample windows.
- The fractional transform is discretised as chirp-multiply -> FFT ->
  chirp-multiply on a symmetric grid (exact at theta = pi/2, i.e. the ordinary
  DFT). The paper does not state its discretisation. ``frft`` is a standalone
  function so a different scheme can be dropped in without touching the model.
- Eq. 9 writes the transition as ``A_bar = delta * A``; this uses the standard
  selective-SSM (S6) zero-order hold ``A_bar = exp(delta * A)`` with ``A``
  negative, which is what makes the recurrence stable.
- Eq. 13's feature map ``phi(x) = exp(W x)`` is clamped before exponentiation to
  keep linear attention finite in fp16/bf16.
- SCA is applied across electrodes at each patch position, which keeps the
  temporal axis intact; the paper's notation (o_c in R^d) is ambiguous here.

Input is ``(B, C, T)`` raw EEG. Nothing in the model is tied to a fixed channel
count -- the patch embedding is shared across channels and SCA pools over them --
which is the paper's claim of montage independence.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor

_THETA_EPS = 1e-2  # keep theta away from 0 and pi, where cot(theta) diverges


# --------------------------------------------------------------------------- #
# Fractional Fourier transform
# --------------------------------------------------------------------------- #
def hermite_basis(t: Tensor, order: int) -> Tensor:
    r"""Physicists' Hermite polynomials :math:`H_0..H_{order}` evaluated at ``t``.

    Uses the recurrence :math:`H_{n+1}(t) = 2t H_n(t) - 2n H_{n-1}(t)`.

    Args:
        t (torch.Tensor): sample grid, shape ``(N,)``.
        order (int): highest polynomial order ``K``.

    Returns:
        torch.Tensor: shape ``(K + 1, N)``.
    """
    polys = [torch.ones_like(t)]
    if order >= 1:
        polys.append(2.0 * t)
    for n in range(1, order):
        polys.append(2.0 * t * polys[n] - 2.0 * n * polys[n - 1])
    return torch.stack(polys, dim=0)


def frft(x: Tensor, theta: Tensor, dim: int = -2) -> Tensor:
    r"""Fractional Fourier transform along ``dim`` with per-feature order ``theta``.

    Discretises Eq. 1 of the paper as chirp -> FFT -> chirp on a symmetric grid
    with spacing :math:`1/\sqrt{N}`, so that ``theta = pi / 2`` reproduces the
    ordinary centred DFT. ``theta`` broadcasts against the trailing feature axis.

    Args:
        x (torch.Tensor): real or complex input, transform axis at ``dim``.
        theta (torch.Tensor): fractional orders in ``(0, pi)``, shape ``(D,)``.
        dim (int): transform axis. (default: :obj:`-2`)

    Returns:
        torch.Tensor: complex tensor shaped like ``x``.
    """
    if dim != -2:
        x = x.transpose(dim, -2)

    n = x.shape[-2]
    xc = x if x.is_complex() else x.to(torch.complex64)

    theta = theta.clamp(_THETA_EPS, math.pi - _THETA_EPS)
    cot = (torch.cos(theta) / torch.sin(theta)).to(x.real.dtype if x.is_complex() else x.dtype)

    # Symmetric time/frequency grid; spacing 1/sqrt(N) makes the transform
    # unitary-ish and self-consistent between the two chirps.
    grid = (torch.arange(n, device=x.device, dtype=cot.dtype) - n // 2) / math.sqrt(n)
    quad = (grid ** 2).unsqueeze(-1)  # (N, 1)

    chirp = torch.exp(1j * math.pi * quad * cot)  # (N, D)
    amp = torch.sqrt(1.0 - 1j * cot.to(torch.complex64))  # A_theta

    shifted = torch.fft.ifftshift(xc * chirp, dim=-2)
    spectrum = torch.fft.fftshift(torch.fft.fft(shifted, dim=-2), dim=-2)
    out = amp * chirp * spectrum

    if dim != -2:
        out = out.transpose(dim, -2)
    return out


class ImplicitWindow(nn.Module):
    r"""Implicit MLP generating the frame window kernel :math:`\Phi` (Eq. 4).

    :math:`\Phi_{j,k} = \big(\sum_i w_{i,k} e^{-\mathrm{i}(b_{i,k} t_j + c_{i,k})}\big)
    \cdot \big(\sum_n a_{n,k} H_n(t_j)\big)`, with every coefficient learnable.
    The Hermite factor injects localised oscillatory priors; the complex
    sinusoids keep the kernel smooth and quasi-periodic.

    Args:
        d_model (int): number of feature channels ``k``.
        num_sinusoids (int): number of sinusoidal terms ``M``. (default: :obj:`4`)
        hermite_order (int): highest Hermite order ``K``. (default: :obj:`4`)
    """

    def __init__(self, d_model: int, num_sinusoids: int = 4, hermite_order: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_sinusoids = num_sinusoids
        self.hermite_order = hermite_order

        self.w = nn.Parameter(torch.randn(num_sinusoids, d_model) / math.sqrt(num_sinusoids))
        self.b = nn.Parameter(torch.randn(num_sinusoids, d_model))
        self.c = nn.Parameter(torch.zeros(num_sinusoids, d_model))
        self.a = nn.Parameter(torch.randn(hermite_order + 1, d_model) / math.sqrt(hermite_order + 1))

    def forward(self, n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return the complex window kernel, shape ``(n, d_model)``."""
        t = torch.linspace(-1.0, 1.0, n, device=device, dtype=dtype)

        phase = self.b.unsqueeze(1) * t.view(1, n, 1) + self.c.unsqueeze(1)  # (M, n, 1)
        sinusoid = (self.w.unsqueeze(1) * torch.exp(-1j * phase)).sum(dim=0)  # (n, d)

        herm = hermite_basis(t, self.hermite_order)  # (K+1, n)
        envelope = torch.einsum("kn,kd->nd", herm, self.a)  # (n, d)

        return sinusoid * envelope


class FrNFO(nn.Module):
    r"""Fractional neural frame operator (Eqs. 3-5).

    Applies :math:`\hat{X}_{:,k} = e^{-\mathrm{i}\pi\omega^2\cot\theta_k}\odot
    F_{\theta_k}(X_{:,k})\odot F_{\theta_k}(\Psi_{:,k})` and splits the complex
    result into amplitude and phase streams.

    Args:
        d_model (int): feature dimension.
        num_sinusoids (int): sinusoidal terms in the implicit window. (default: :obj:`4`)
        hermite_order (int): Hermite order in the implicit window. (default: :obj:`4`)
    """

    def __init__(self, d_model: int, num_sinusoids: int = 4, hermite_order: int = 4):
        super().__init__()
        self.d_model = d_model
        self.window = ImplicitWindow(d_model, num_sinusoids, hermite_order)
        # theta = pi * sigmoid(raw) keeps the learnable order inside (0, pi).
        self.theta_raw = nn.Parameter(torch.zeros(d_model))

    @property
    def theta(self) -> Tensor:
        return math.pi * torch.sigmoid(self.theta_raw)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """``x``: ``(B, C, N, D)`` real -> ``(amplitude, phase)``, both ``(B, C, N, D)``."""
        n = x.shape[-2]
        theta = self.theta

        psi = self.window(n, x.device, x.dtype)  # (N, D) complex
        fx = frft(x, theta, dim=-2)
        fpsi = frft(psi, theta, dim=-2)

        cot = torch.cos(theta.clamp(_THETA_EPS, math.pi - _THETA_EPS)) / torch.sin(
            theta.clamp(_THETA_EPS, math.pi - _THETA_EPS)
        )
        omega = (torch.arange(n, device=x.device, dtype=x.dtype) - n // 2) / math.sqrt(n)
        adjust = torch.exp(-1j * math.pi * (omega ** 2).unsqueeze(-1) * cot)  # (N, D)

        out = adjust * fx * fpsi
        # |z| via the squared modulus plus an epsilon: torch.abs / torch.angle both
        # have undefined gradients at z = 0, which a zero-padded channel would hit.
        amplitude = (out.real ** 2 + out.imag ** 2 + 1e-12).sqrt()
        phase = torch.atan2(out.imag, out.real + 1e-12)
        return amplitude, phase


# --------------------------------------------------------------------------- #
# Amplitude-phase cross-encoding
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """Root-mean-square layer norm (used at every APCE entry point)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def selective_scan(x: Tensor, a_bar: Tensor, b_bar: Tensor, c: Tensor) -> Tensor:
    r"""Sequential selective-SSM scan (Eq. 10).

    ``h_t = a_bar_t * h_{t-1} + b_bar_t * x_t``, ``y_t = <c_t, h_t>``.

    Args:
        x (torch.Tensor): ``(B, L, E)`` input sequence.
        a_bar (torch.Tensor): ``(B, L, E, S)`` discretised transitions.
        b_bar (torch.Tensor): ``(B, L, E, S)`` discretised input map.
        c (torch.Tensor): ``(B, L, S)`` output map.

    Returns:
        torch.Tensor: ``(B, L, E)``.

    The loop is over patches (``L = T // patch_size``, typically 10-40), so a
    Python-level scan costs little; no custom CUDA kernel is needed.
    """
    b, l, e, s = a_bar.shape
    h = x.new_zeros(b, e, s)
    ys = []
    for t in range(l):
        h = a_bar[:, t] * h + b_bar[:, t] * x[:, t].unsqueeze(-1)
        ys.append(torch.einsum("bes,bs->be", h, c[:, t]))
    return torch.stack(ys, dim=1)


class BidirectionalCrossSSM(nn.Module):
    r"""One bidirectional cross state-space block (Eqs. 6-11).

    The *driven* stream supplies ``x`` and the gate ``z``; the *context* stream
    supplies the selective parameters ``B`` and ``C``, which is what couples
    amplitude to phase.

    Args:
        d_model (int): input/output width ``D``.
        d_inner (int): latent SSM width ``E``.
        d_state (int): number of SSM states ``S``.
        conv_kernel (int): causal/anti-causal convolution width. (default: :obj:`4`)
    """

    def __init__(self, d_model: int, d_inner: int, d_state: int, conv_kernel: int = 4):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        self.conv_kernel = conv_kernel

        self.w_x = nn.Linear(d_model, d_inner, bias=False)
        self.w_z = nn.Linear(d_model, d_inner, bias=False)
        self.w_b = nn.Linear(d_model, d_state, bias=False)
        self.w_c = nn.Linear(d_model, d_state, bias=False)
        self.w_delta = nn.Linear(d_inner, d_inner, bias=True)

        self.conv_fwd = nn.Conv1d(d_inner, d_inner, conv_kernel, groups=d_inner)
        self.conv_bwd = nn.Conv1d(d_inner, d_inner, conv_kernel, groups=d_inner)

        # A is shared and direction-agnostic; parametrised as -exp(log A) < 0.
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(d_inner, 1)
        )
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def _directional_conv(self, x: Tensor, backward: bool) -> Tensor:
        """Causal (or anti-causal) depthwise conv + SiLU, Eq. 7."""
        seq = x.flip(1) if backward else x
        seq = seq.transpose(1, 2)  # (B, E, L)
        seq = F.pad(seq, (self.conv_kernel - 1, 0))
        conv = self.conv_bwd if backward else self.conv_fwd
        out = F.silu(conv(seq)).transpose(1, 2)
        return out.flip(1) if backward else out

    def forward(self, driven: Tensor, context: Tensor) -> Tensor:
        """``driven``/``context``: ``(B, L, D)`` -> ``(B, L, D)``."""
        x_proj = self.w_x(driven)
        z = self.w_z(driven)

        b_ctx = self.w_b(context)  # (B, L, S)
        c_ctx = self.w_c(context)  # (B, L, S)
        a = -torch.exp(self.a_log)  # (E, S)

        outputs = []
        for backward in (False, True):
            x_o = self._directional_conv(x_proj, backward)  # (B, L, E)
            delta = F.softplus(self.w_delta(x_o))  # (B, L, E)
            a_bar = torch.exp(delta.unsqueeze(-1) * a)  # (B, L, E, S)
            b_bar = delta.unsqueeze(-1) * b_ctx.unsqueeze(2)  # (B, L, E, S)
            outputs.append(selective_scan(x_o, a_bar, b_bar, c_ctx))

        y = (outputs[0] + outputs[1]) * F.silu(z)
        return self.out_proj(y)


class APCE(nn.Module):
    r"""Amplitude-phase cross-encoding (Eq. 12).

    Phase is encoded first with amplitude as context; the resulting
    phase-informative sequence then becomes the context for the amplitude
    block, and the amplitude stream is added back as a residual.

    Args:
        d_model (int): feature width.
        d_inner (int): latent SSM width.
        d_state (int): number of SSM states.
    """

    def __init__(self, d_model: int, d_inner: int, d_state: int):
        super().__init__()
        self.norm_amp = RMSNorm(d_model)
        self.norm_pha = RMSNorm(d_model)
        self.phase_ssm = BidirectionalCrossSSM(d_model, d_inner, d_state)
        self.norm_mid = RMSNorm(d_model)
        self.amp_ssm = BidirectionalCrossSSM(d_model, d_inner, d_state)

    def forward(self, amplitude: Tensor, phase: Tensor) -> Tensor:
        """``(B, C, N, D)`` x2 -> ``(B, C, N, D)``."""
        b, c, n, d = amplitude.shape
        amp = self.norm_amp(amplitude).reshape(b * c, n, d)
        pha = self.norm_pha(phase).reshape(b * c, n, d)

        y_phase = self.phase_ssm(pha, amp)  # phase driven, amplitude context
        y_amp = self.amp_ssm(amp, self.norm_mid(y_phase))  # roles swapped

        return (y_amp + amp).reshape(b, c, n, d)


# --------------------------------------------------------------------------- #
# Spatial correlation aggregation
# --------------------------------------------------------------------------- #
class SpatialCorrelationAggregation(nn.Module):
    r"""Gated linear attention across electrodes (Eq. 13).

    For every patch position, linear attention runs over the channel axis at
    linear cost in the electrode count, and its output is gated by
    ``sigmoid(RMSNorm(DepthwiseConv2d_{3x3}(X)))`` so local (channel, time)
    neighbourhoods modulate the global aggregation.

    Args:
        d_model (int): feature width.
        clamp (float): pre-exponential clamp for the feature map. (default: :obj:`8.0`)
    """

    def __init__(self, d_model: int, clamp: float = 8.0):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.phi = nn.Linear(d_model, d_model, bias=False)
        self.gate_conv = nn.Conv2d(d_model, d_model, 3, padding=1, groups=d_model)
        self.gate_norm = RMSNorm(d_model)
        self.clamp = clamp

    def _feature_map(self, x: Tensor) -> Tensor:
        """``phi(x) = exp(W x)``, clamped so linear attention stays finite."""
        return torch.exp(self.phi(x).clamp(-self.clamp, self.clamp))

    def forward(self, x: Tensor) -> Tensor:
        """``(B, C, N, D)`` -> ``(B, C, N, D)``."""
        q = self._feature_map(self.q_proj(x))
        k = self._feature_map(self.k_proj(x))
        v = self.v_proj(x)

        # Sum over the channel axis once, then reuse for every query channel.
        kv = torch.einsum("bcnd,bcne->bnde", k, v)
        num = torch.einsum("bcnd,bnde->bcne", q, kv)
        den = torch.einsum("bcnd,bnd->bcn", q, k.sum(dim=1)).unsqueeze(-1)
        attn = num / den.clamp_min(1e-6)

        gate = self.gate_conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return attn * torch.sigmoid(self.gate_norm(gate))


class FAPEXBlock(nn.Module):
    """One backbone layer: FrNFO -> APCE -> SCA, with a residual around SCA."""

    def __init__(self, d_model: int, d_inner: int, d_state: int,
                 num_sinusoids: int, hermite_order: int):
        super().__init__()
        self.frnfo = FrNFO(d_model, num_sinusoids, hermite_order)
        self.apce = APCE(d_model, d_inner, d_state)
        self.sca = SpatialCorrelationAggregation(d_model)
        self.norm = RMSNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        amplitude, phase = self.frnfo(x)
        mixed = self.apce(amplitude, phase)
        return self.norm(mixed + self.sca(mixed))


class FAPEX(nn.Module):
    r'''
    FAPEX (Zheng et al., NeurIPS 2025): fractional time-frequency decomposition,
    amplitude-phase cross-encoding over bidirectional state-space blocks, and
    linear attention across electrodes.

    Reconstructed from the paper text; see the module docstring for the list of
    inferred choices. The model is channel-count agnostic, so the same weights
    accept any montage.

    .. code-block:: yaml

        model:
          name: fapex
          num_classes: 1
          kwargs:
            patch_size: 32     # tau; T must be divisible by it
            d_model: 64
            depth: 4

    .. code-block:: python

        model = FAPEX(num_classes=1, patch_size=32, d_model=64, depth=2)
        logits = model(torch.randn(4, 18, 640))  # (4, 1)

    Args:
        num_classes (int): output logits; use :obj:`1` for a single binary logit. (default: :obj:`2`)
        patch_size (int): patch length :math:`\tau` in samples. (default: :obj:`32`)
        d_model (int): embedding width :math:`d_{model}`. (default: :obj:`64`)
        d_inner (int): latent SSM width :math:`E`; defaults to :obj:`d_model`. (default: :obj:`None`)
        d_state (int): number of SSM states :math:`S`. (default: :obj:`16`)
        depth (int): number of FrNFO/APCE/SCA layers. (default: :obj:`4`)
        num_sinusoids (int): sinusoidal terms :math:`M` in the implicit window. (default: :obj:`4`)
        hermite_order (int): Hermite order :math:`K` in the implicit window. (default: :obj:`4`)
        dropout (float): dropout before the classifier. (default: :obj:`0.1`)
    '''

    def __init__(self,
                 num_classes: int = 2,
                 patch_size: int = 32,
                 d_model: int = 64,
                 d_inner: Optional[int] = None,
                 d_state: int = 16,
                 depth: int = 4,
                 num_sinusoids: int = 4,
                 hermite_order: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        if patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {patch_size}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.patch_size = patch_size
        self.d_model = d_model
        self.d_inner = d_inner or d_model
        self.d_state = d_state
        self.num_classes = num_classes

        # Channel-shared patch embedding: W in R^{d_model x tau} (paper Sec. 2).
        self.patch_embed = nn.Linear(patch_size, d_model)

        self.blocks = nn.ModuleList([
            FAPEXBlock(d_model, self.d_inner, d_state, num_sinusoids, hermite_order)
            for _ in range(depth)
        ])

        self.head_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, num_classes)

    def embed(self, x: Tensor) -> Tensor:
        """Return the pooled backbone embedding, ``(B, d_model)``.

        Exposed separately so the projection head of the paper's self-supervised
        variant can be attached without re-running the classifier.
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T) input, got shape {tuple(x.shape)}")
        b, c, t = x.shape
        if t < self.patch_size:
            raise ValueError(
                f"window length {t} is shorter than patch_size {self.patch_size}"
            )

        n = t // self.patch_size
        patches = x[..., : n * self.patch_size].reshape(b, c, n, self.patch_size)
        h = self.patch_embed(patches)  # (B, C, N, D)

        for block in self.blocks:
            h = block(h)

        return self.head_norm(h.mean(dim=(1, 2)))  # pool electrodes and patches

    def forward(self, x: Tensor) -> Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG window, shape :obj:`(B, C, T)`. ``T`` is truncated
                to a whole number of patches.

        Returns:
            torch.Tensor: logits, shape :obj:`(B, num_classes)`.
        '''
        return self.head(self.dropout(self.embed(x)))


# ---- seizure_pred registry glue ----
from seizure_pred.core.config import ModelConfig
from seizure_pred.training.registries import MODELS


@MODELS.register("fapex", help="FAPEX: fractional amplitude-phase expressor (NeurIPS 2025).")
def build_fapex(cfg: ModelConfig) -> nn.Module:
    kw = dict(getattr(cfg, "kwargs", {}) or {})
    return FAPEX(
        num_classes=int(getattr(cfg, "num_classes", 2)),
        patch_size=int(kw.get("patch_size", 32)),
        d_model=int(kw.get("d_model", 64)),
        d_inner=int(kw["d_inner"]) if kw.get("d_inner") else None,
        d_state=int(kw.get("d_state", 16)),
        depth=int(kw.get("depth", 4)),
        num_sinusoids=int(kw.get("num_sinusoids", 4)),
        hermite_order=int(kw.get("hermite_order", 4)),
        dropout=float(kw.get("dropout", 0.1)),
    )
