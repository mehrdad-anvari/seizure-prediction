from __future__ import annotations

r"""MD-ResCapsNet: multidimensional residual capsule network for seizure prediction.

- Paper: Xi Y, Lan Z, Meng T, Chen Y, Cao J, Zhang L. Epileptic seizure prediction
  based on a residual capsule network with multidimensional electroencephalography
  feature fusion. Biomedical Signal Processing and Control, 114 (2026) 109370.
- Related project: none published.

Reconstruction from the paper text. The three stages of Fig. 2:

1. **CSP spatial feature enhancement** (Eqs. 1-8). Analytically computed spatial
   filters, not learned: covariances are trace-normalised, summed, whitened with
   Tikhonov regularisation, and a generalised eigenproblem yields the projection
   ``W_CSP``. Use :func:`compute_csp_filters` on your training split and pass the
   result as ``csp_filters``; without it the stage is a pass-through, since
   fitting CSP inside the model would leak label information across folds.
2. **Multidimensional feature extraction** (Eqs. 9-13). STFT to a
   channel-averaged time-frequency image, a 3x3 stem without pooling, then four
   stages of two residual blocks each (64, 128, 256, 512 channels) where every
   block carries a squeeze-and-excitation module and a 7x7 spatial attention map.
3. **Capsule prediction** (Eq. 14). A 1x1 convolution forms primary capsules,
   squashed to unit-bounded norms, and dynamic routing aggregates them into one
   capsule per class.

Inferred, because the paper does not state them:

- STFT hop size (50% overlap here), the SE reduction ratio (16), the capsule
  geometry (8 primary capsule types of dimension 8, class capsules of dimension
  16) and the routing iteration count (3, the value from Sabour et al.).
- The paper says "at each stage, the first residual block applies a stride of
  two", so stage 1 downsamples too -- unlike a standard ResNet. That reading is
  implemented literally; set ``stem_stride_stage1=False`` for ResNet behaviour.
- Class scores come from capsule norms, which are already in ``[0, 1]``. To
  satisfy this library's logits contract they are mapped back through a logit
  transform, so ``bce_logits`` and ``focal`` work unchanged. For the paper's
  objective use the ``capsule_margin`` loss, which inverts the transform.

Input is ``(B, C, T)`` raw EEG; the STFT front-end runs inside the model, so
``sfreq`` matters -- pass it via ``ModelConfig.sfreq`` or ``kwargs['sfreq']``.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# CSP spatial filters (Eqs. 1-8)
# --------------------------------------------------------------------------- #
def compute_csp_filters(preictal: Tensor, interictal: Tensor, *, reg: float = 1e-3,
                        n_components: Optional[int] = None) -> Tensor:
    r"""Fit the CSP projection of Eqs. 1-8 from labelled segments.

    Fit this on training data only, then hand the result to
    :class:`MD_ResCapsNet` as ``csp_filters``.

    Args:
        preictal (torch.Tensor): preictal segments, ``(N_pre, C, T)``.
        interictal (torch.Tensor): interictal segments, ``(N_int, C, T)``.
        reg (float): Tikhonov coefficient :math:`\lambda` of Eq. 4. (default: :obj:`1e-3`)
        n_components (int, optional): number of filters to keep, taken from the
            extremes of the eigenvalue spectrum. Keeps all channels if
            :obj:`None`. (default: :obj:`None`)

    Returns:
        torch.Tensor: projection matrix ``(n_components, C)``, ready to be applied
        as :math:`Z = W_{CSP}^\top X`.
    """
    def _cov(x: Tensor) -> Tensor:
        x = x.to(torch.float64)
        prod = x @ x.transpose(-1, -2)  # (N, C, C)
        trace = prod.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-12)
        return (prod / trace[:, None, None]).mean(dim=0)

    c_pre, c_int = _cov(preictal), _cov(interictal)
    c_sum = c_pre + c_int  # Eq. 2

    evals, evecs = torch.linalg.eigh(c_sum)  # Eq. 3
    whiten = torch.diag((evals + reg).clamp_min(1e-12).rsqrt()) @ evecs.T  # Eq. 4

    s_pre = whiten @ c_pre @ whiten.T  # Eq. 5
    s_int = whiten @ c_int @ whiten.T
    # Eq. 6: generalised eigenproblem, solved symmetrically in the whitened space.
    _, w = torch.linalg.eigh(s_pre - s_int)

    filters = (whiten.T @ w).T  # Eq. 7, transposed to (n_filters, C)
    if n_components is not None and n_components < filters.shape[0]:
        half = n_components // 2
        keep = list(range(half)) + list(range(filters.shape[0] - (n_components - half), filters.shape[0]))
        filters = filters[keep]
    return filters.to(torch.float32)


# --------------------------------------------------------------------------- #
# Attention blocks
# --------------------------------------------------------------------------- #
class SEBlock(nn.Module):
    r"""Squeeze-and-excitation channel recalibration (Eqs. 10-12).

    Args:
        channels (int): number of feature channels ``M``.
        reduction (int): bottleneck ratio of the two fully connected layers. (default: :obj:`16`)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: Tensor) -> Tensor:
        z = x.mean(dim=(2, 3))  # squeeze, Eq. 10
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(z))))  # Eq. 11
        return x * s[:, :, None, None]  # Eq. 12


class SpatialAttention(nn.Module):
    r"""Spatial attention map from pooled channel statistics (Eq. 13).

    Args:
        kernel_size (int): convolution width applied to the pooled maps. (default: :obj:`7`)
    """

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        pooled = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(pooled))


class SESAResidualBlock(nn.Module):
    r"""Residual block with two 3x3 convolutions followed by SE and SA (Fig. 3).

    Args:
        in_channels (int): input channels.
        out_channels (int): output channels.
        stride (int): stride of the first convolution. (default: :obj:`1`)
        reduction (int): SE bottleneck ratio. (default: :obj:`16`)
        sa_kernel (int): spatial-attention kernel size. (default: :obj:`7`)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 reduction: int = 16, sa_kernel: int = 7):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels, reduction)
        self.sa = SpatialAttention(sa_kernel)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.sa(self.se(out))
        return F.relu(out + identity)


# --------------------------------------------------------------------------- #
# Capsules
# --------------------------------------------------------------------------- #
def squash(s: Tensor, dim: int = -1, eps: float = 1e-8) -> Tensor:
    r"""Capsule scaling of Eq. 14: :math:`v = \frac{\|s\|^2}{1+\|s\|^2}\frac{s}{\|s\|}`."""
    sq_norm = s.pow(2).sum(dim=dim, keepdim=True)
    norm = (sq_norm + eps).sqrt()
    return (sq_norm / (1.0 + sq_norm)) * (s / norm)


class PrimaryCapsules(nn.Module):
    r"""Primary capsule layer: a 1x1 convolution reshaped into squashed capsules.

    Args:
        in_channels (int): backbone output channels.
        num_capsules (int): number of capsule types.
        capsule_dim (int): dimension of each primary capsule.
    """

    def __init__(self, in_channels: int, num_capsules: int, capsule_dim: int):
        super().__init__()
        self.num_capsules = num_capsules
        self.capsule_dim = capsule_dim
        self.conv = nn.Conv2d(in_channels, num_capsules * capsule_dim, 1)

    def forward(self, x: Tensor) -> Tensor:
        """``(B, C, H, W)`` -> ``(B, num_capsules * H * W, capsule_dim)``."""
        u = self.conv(x)
        b, _, h, w = u.shape
        u = u.view(b, self.num_capsules, self.capsule_dim, h * w)
        u = u.permute(0, 1, 3, 2).reshape(b, self.num_capsules * h * w, self.capsule_dim)
        return squash(u)


class RoutingCapsules(nn.Module):
    r"""Class capsules produced by dynamic routing (Sec. 2.2.3).

    Coupling coefficients start at zero, are softmaxed over output capsules, and
    are refined by agreement between predictions and outputs.

    Args:
        num_in_capsules (int): number of primary capsules.
        in_dim (int): primary capsule dimension.
        num_out_capsules (int): number of class capsules.
        out_dim (int): class capsule dimension.
        iterations (int): routing iterations. (default: :obj:`3`)
    """

    def __init__(self, num_in_capsules: int, in_dim: int, num_out_capsules: int,
                 out_dim: int, iterations: int = 3):
        super().__init__()
        self.iterations = iterations
        self.num_out_capsules = num_out_capsules
        self.weight = nn.Parameter(
            0.01 * torch.randn(num_out_capsules, num_in_capsules, out_dim, in_dim)
        )

    def forward(self, u: Tensor) -> Tensor:
        """``(B, N_in, in_dim)`` -> ``(B, num_out_capsules, out_dim)``."""
        # u_hat[b, j, i, :] = W[j, i] @ u[b, i]
        u_hat = torch.einsum("jiod,bid->bjio", self.weight, u)
        logits = u_hat.new_zeros(u_hat.shape[:3])  # b_ij, initialised to zero

        u_hat_detached = u_hat.detach()
        for it in range(self.iterations):
            coupling = F.softmax(logits, dim=1)  # over output capsules
            # Only the final iteration backpropagates through u_hat (standard
            # practice: earlier iterations are a fixed-point search).
            target = u_hat if it == self.iterations - 1 else u_hat_detached
            v = squash(torch.einsum("bji,bjio->bjo", coupling, target))
            if it != self.iterations - 1:
                logits = logits + torch.einsum("bjio,bjo->bji", u_hat_detached, v)
        return v


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class MD_ResCapsNet(nn.Module):
    r'''
    MD-ResCapsNet (Xi et al., BSPC 2026): CSP-enhanced EEG, STFT time-frequency
    images, an SE-SA residual backbone, and a capsule head with dynamic routing.

    Reconstructed from the paper text; see the module docstring for inferred
    choices.

    .. code-block:: yaml

        model:
          name: md_rescapsnet
          num_classes: 1
          sfreq: 128
          kwargs:
            f_max: 30.0
            stage_channels: [64, 128, 256, 512]

    .. code-block:: python

        model = MD_ResCapsNet(num_classes=1, sfreq=128, chunk_size=640)
        logits = model(torch.randn(4, 18, 640))  # (4, 1)

    Args:
        num_classes (int): output logits; use :obj:`1` for a single binary logit. (default: :obj:`2`)
        in_channels (int): number of EEG channels. (default: :obj:`18`)
        chunk_size (int): window length ``T`` in samples, used for the shape probe. (default: :obj:`640`)
        sfreq (float): sampling rate in Hz, needed to place the STFT frequency cut. (default: :obj:`128.0`)
        f_max (float): highest retained STFT frequency in Hz. (default: :obj:`30.0`)
        stft_window (int, optional): STFT window length in samples; defaults to one
            second (``round(sfreq)``). (default: :obj:`None`)
        stft_hop (int, optional): STFT hop; defaults to half the window. (default: :obj:`None`)
        stem_channels (int): channels of the 3x3 stem. (default: :obj:`64`)
        stage_channels (tuple): channels per residual stage. (default: :obj:`(64, 128, 256, 512)`)
        blocks_per_stage (int): residual blocks per stage. (default: :obj:`2`)
        stem_stride_stage1 (bool): whether stage 1 also downsamples, as the paper
            states. (default: :obj:`True`)
        se_reduction (int): SE bottleneck ratio. (default: :obj:`16`)
        num_primary_capsules (int): primary capsule types. (default: :obj:`8`)
        primary_capsule_dim (int): primary capsule dimension. (default: :obj:`8`)
        class_capsule_dim (int): class capsule dimension. (default: :obj:`16`)
        routing_iterations (int): dynamic routing iterations. (default: :obj:`3`)
        dropout (float): dropout after the stem. (default: :obj:`0.1`)
        csp_filters (torch.Tensor, optional): projection from
            :func:`compute_csp_filters`, shape ``(n_filters, in_channels)``. Without
            it the CSP stage is skipped. (default: :obj:`None`)
    '''

    def __init__(self,
                 num_classes: int = 2,
                 in_channels: int = 18,
                 chunk_size: int = 640,
                 sfreq: float = 128.0,
                 f_max: float = 30.0,
                 stft_window: Optional[int] = None,
                 stft_hop: Optional[int] = None,
                 stem_channels: int = 64,
                 stage_channels: Tuple[int, ...] = (64, 128, 256, 512),
                 blocks_per_stage: int = 2,
                 stem_stride_stage1: bool = True,
                 se_reduction: int = 16,
                 num_primary_capsules: int = 8,
                 primary_capsule_dim: int = 8,
                 class_capsule_dim: int = 16,
                 routing_iterations: int = 3,
                 dropout: float = 0.1,
                 csp_filters: Optional[Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.chunk_size = chunk_size
        self.sfreq = float(sfreq)
        self.f_max = float(f_max)

        self.n_fft = int(stft_window) if stft_window else max(16, int(round(self.sfreq)))
        self.hop = int(stft_hop) if stft_hop else max(1, self.n_fft // 2)
        if self.n_fft > chunk_size:
            raise ValueError(
                f"stft_window ({self.n_fft}) exceeds the window length ({chunk_size})"
            )
        # Highest bin at or below f_max; bin k sits at k * sfreq / n_fft.
        self.n_freq = min(self.n_fft // 2 + 1,
                          int(math.floor(self.f_max * self.n_fft / self.sfreq)) + 1)
        self.register_buffer("stft_win", torch.hann_window(self.n_fft), persistent=False)

        if csp_filters is not None:
            filt = torch.as_tensor(csp_filters, dtype=torch.float32)
            if filt.ndim != 2 or filt.shape[1] != in_channels:
                raise ValueError(
                    f"csp_filters must have shape (n_filters, {in_channels}), got {tuple(filt.shape)}"
                )
            self.register_buffer("csp_filters", filt)
        else:
            self.register_buffer("csp_filters", None)

        self.stem = nn.Sequential(
            nn.Conv2d(1, stem_channels, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        stages = []
        prev = stem_channels
        for idx, channels in enumerate(stage_channels):
            first_stride = 2 if (idx > 0 or stem_stride_stage1) else 1
            for block in range(blocks_per_stage):
                stages.append(
                    SESAResidualBlock(
                        prev, channels,
                        stride=first_stride if block == 0 else 1,
                        reduction=se_reduction,
                    )
                )
                prev = channels
        self.backbone = nn.Sequential(*stages)
        self.backbone_channels = prev

        feat_h, feat_w = self._probe_backbone_shape()
        self.primary_capsules = PrimaryCapsules(prev, num_primary_capsules, primary_capsule_dim)
        # Always at least two class capsules (interictal / preictal), as in the
        # paper: with a single output capsule the routing softmax is degenerate.
        self.class_capsules = RoutingCapsules(
            num_in_capsules=num_primary_capsules * feat_h * feat_w,
            in_dim=primary_capsule_dim,
            num_out_capsules=max(2, num_classes),
            out_dim=class_capsule_dim,
            iterations=routing_iterations,
        )

    # -- front-end ---------------------------------------------------------- #
    def spectrogram(self, x: Tensor) -> Tensor:
        """CSP projection then channel-averaged STFT magnitude (Eqs. 8-9).

        Returns a single-channel time-frequency image, ``(B, 1, n_freq, frames)``.
        """
        if self.csp_filters is not None:
            x = torch.einsum("fc,bct->bft", self.csp_filters, x)  # Eq. 8

        b, c, t = x.shape
        spec = torch.stft(
            x.reshape(b * c, t),
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.stft_win.to(x.dtype),
            center=True,
            return_complex=True,
        )
        mag = spec.abs()[:, : self.n_freq, :]  # crop to 0..f_max
        mag = mag.reshape(b, c, self.n_freq, -1).mean(dim=1)  # average channels
        return mag.unsqueeze(1)

    def _probe_backbone_shape(self) -> Tuple[int, int]:
        """Spatial size of the backbone output, measured on a zero window."""
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                mock = torch.zeros(1, self.in_channels, self.chunk_size)
                out = self.backbone(self.stem(self.spectrogram(mock)))
                return int(out.shape[2]), int(out.shape[3])
        finally:
            self.train(was_training)

    # -- forward ------------------------------------------------------------ #
    def capsule_lengths(self, x: Tensor) -> Tensor:
        """Class capsule norms in ``[0, 1]``, ``(B, num_out_capsules)``."""
        h = self.backbone(self.stem(self.spectrogram(x)))
        v = self.class_capsules(self.primary_capsules(h))
        return v.norm(dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG window, shape :obj:`(B, C, T)`.

        Returns:
            torch.Tensor: logits, shape :obj:`(B, num_classes)`. Capsule norms are
            converted with a logit transform so standard logit-based losses apply;
            recover the norms as :obj:`torch.sigmoid(logits)`. With
            ``num_classes=1`` the preictal capsule (index 1) supplies the logit.
        '''
        lengths = self.capsule_lengths(x).clamp(1e-6, 1.0 - 1e-6)
        logits = torch.log(lengths) - torch.log1p(-lengths)
        if self.num_classes == 1:
            return logits[:, 1:2]
        return logits[:, : self.num_classes]


# ---- seizure_pred registry glue ----
from seizure_pred.core.config import ModelConfig
from seizure_pred.training.registries import MODELS


@MODELS.register("md_rescapsnet", help="MD-ResCapsNet: CSP + SE-SA ResNet + capsule routing (BSPC 2026).")
def build_md_rescapsnet(cfg: ModelConfig) -> nn.Module:
    kw = dict(getattr(cfg, "kwargs", {}) or {})
    in_channels = int(cfg.in_channels or kw.get("in_channels", 18))
    sfreq = float(getattr(cfg, "sfreq", None) or kw.get("sfreq", 128.0))
    stage_channels = tuple(int(c) for c in kw.get("stage_channels", (64, 128, 256, 512)))
    return MD_ResCapsNet(
        num_classes=int(getattr(cfg, "num_classes", 2)),
        in_channels=in_channels,
        chunk_size=int(kw.get("chunk_size", kw.get("seq_len", 640))),
        sfreq=sfreq,
        f_max=float(kw.get("f_max", 30.0)),
        stft_window=int(kw["stft_window"]) if kw.get("stft_window") else None,
        stft_hop=int(kw["stft_hop"]) if kw.get("stft_hop") else None,
        stem_channels=int(kw.get("stem_channels", 64)),
        stage_channels=stage_channels,
        blocks_per_stage=int(kw.get("blocks_per_stage", 2)),
        stem_stride_stage1=bool(kw.get("stem_stride_stage1", True)),
        se_reduction=int(kw.get("se_reduction", 16)),
        num_primary_capsules=int(kw.get("num_primary_capsules", 8)),
        primary_capsule_dim=int(kw.get("primary_capsule_dim", 8)),
        class_capsule_dim=int(kw.get("class_capsule_dim", 16)),
        routing_iterations=int(kw.get("routing_iterations", 3)),
        dropout=float(kw.get("dropout", 0.1)),
        csp_filters=kw.get("csp_filters"),
    )
