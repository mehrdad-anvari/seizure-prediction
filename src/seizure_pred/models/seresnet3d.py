from __future__ import annotations

r"""3D-SERESNet: squeeze-and-excitation 3D residual network for seizure prediction.

- Paper: Meng L, Zhou L, Zhang W, Xie K. Robust epileptic seizure prediction: A
  3D-SERESNet framework for patient-specific and multi-patient generalization.
  iScience 28, 114171 (December 19, 2025).
- Authors' archive: https://zenodo.org/records/17495041 (referenced by the paper;
  this file is written from the paper text, not from that deposit).

Reconstruction from the STAR Methods. The pipeline:

1. **3D time-frequency volume.** Each channel is z-scored (Eq. 3), transformed by
   STFT (Hann window, 64-sample segments, 50% overlap, 256-point FFT), and the
   per-channel maps are stacked into a volume ``M`` -- 129 frequency bins x 79
   frames x 18 channels for the paper's 10 s windows at 256 Hz.
2. **Stem** (Eq. 4): ``U = AvgPool(LeakyReLU(Conv3d(M)))``.
3. **Three SE-ResNet modules** (Eqs. 5-6). Each residual block is a 1x1x1
   channel-modulating convolution, BN, LeakyReLU, a 1x3x3 convolution over the
   frequency-time plane, BN, LeakyReLU, a 1x1x1 convolution restoring the channel
   count, BN, the skip addition, LeakyReLU, then dropout. Modules 2 and 3 are
   preceded by a convolution + BN layer. Each module ends in SE recalibration.
4. **Head** (Eq. 7): ``LeakyReLU(AdaptiveAvgPool(LeakyReLU(ConvLayer(O))))``
   followed by two fully connected layers.

The paper pairs this with focal loss, which this library already provides as the
``focal`` loss -- no model-side work needed.

Stated by the paper and implemented as such: the 1x1x1 / 1x3x3 / 1x1x1 bottleneck
ordering, BN placement, LeakyReLU throughout, dropout at the end of each residual
block, three modules, the extra conv layer before modules 2 and 3, adaptive
pooling, and two FC layers.

Inferred, because the paper states no channel widths or strides:

- Feature widths (stem 32, modules 32 -> 64 -> 64, head 128) and the bottleneck
  ratio (1/2). The stem width matches module 1's width so that module needs no
  leading convolution, keeping Eq. 5 (``O(1) = SE(ResBlock(U))``) literal.
- Downsampling: an average pool of ``(1, 2, 2)`` in the stem and stride
  ``(1, 2, 2)`` in the conv layers before modules 2 and 3 -- always over the
  frequency-time plane, never across electrodes.
- The stem kernel is ``(3, 3, 3)`` so it mixes electrodes, which is the stated
  purpose of the 3D representation; the residual blocks then keep the paper's
  ``1x3x3`` kernels, which do not.
- Spectrogram values are powers (``|X|^2``), matching SciPy's ``mode='psd'``
  default; set ``log_power=True`` for decibel-like scaling.

Input is ``(B, C, T)`` raw EEG; the STFT front-end runs inside the model.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class SEBlock3d(nn.Module):
    r"""Squeeze-and-excitation over feature channels of a 5D tensor (Fig. 8B).

    Args:
        channels (int): number of feature channels.
        reduction (int): bottleneck ratio of the two FC layers. (default: :obj:`16`)
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: Tensor) -> Tensor:
        z = x.mean(dim=(2, 3, 4))  # global average pooling
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(z))))
        return x * s[:, :, None, None, None]


class ResBlock3d(nn.Module):
    r"""Bottleneck residual block with 1x1x1 / 1x3x3 / 1x1x1 convolutions (Fig. 8A).

    Args:
        channels (int): input and output channels.
        bottleneck_ratio (float): width of the inner 1x3x3 convolution relative to
            ``channels``. (default: :obj:`0.5`)
        dropout (float): dropout applied to the block output. (default: :obj:`0.1`)
        negative_slope (float): LeakyReLU slope. (default: :obj:`0.01`)
    """

    def __init__(self, channels: int, bottleneck_ratio: float = 0.5,
                 dropout: float = 0.1, negative_slope: float = 0.01):
        super().__init__()
        hidden = max(1, int(channels * bottleneck_ratio))
        self.negative_slope = negative_slope

        self.conv_in = nn.Conv3d(channels, hidden, kernel_size=1, bias=False)
        self.bn_in = nn.BatchNorm3d(hidden)
        self.conv_mid = nn.Conv3d(hidden, hidden, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False)
        self.bn_mid = nn.BatchNorm3d(hidden)
        self.conv_out = nn.Conv3d(hidden, channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm3d(channels)
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x: Tensor) -> Tensor:
        out = F.leaky_relu(self.bn_in(self.conv_in(x)), self.negative_slope)
        out = F.leaky_relu(self.bn_mid(self.conv_mid(out)), self.negative_slope)
        out = self.bn_out(self.conv_out(out))
        out = F.leaky_relu(out + x, self.negative_slope)
        return self.dropout(out)


class SEResNetModule(nn.Module):
    r"""One SE-ResNet module: optional conv layer, residual block, SE (Eqs. 5-6).

    Args:
        in_channels (int): input channels.
        out_channels (int): channels after the optional leading conv layer.
        stride (tuple): stride of the leading conv layer. (default: :obj:`(1, 1, 1)`)
        lead_conv (bool): whether the module starts with conv + BN, as modules 2
            and 3 do in Eq. 6. (default: :obj:`True`)
        bottleneck_ratio (float): residual bottleneck width ratio. (default: :obj:`0.5`)
        dropout (float): dropout inside the residual block. (default: :obj:`0.1`)
        se_reduction (int): SE bottleneck ratio. (default: :obj:`16`)
        negative_slope (float): LeakyReLU slope. (default: :obj:`0.01`)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 stride: Tuple[int, int, int] = (1, 1, 1), lead_conv: bool = True,
                 bottleneck_ratio: float = 0.5, dropout: float = 0.1,
                 se_reduction: int = 16, negative_slope: float = 0.01):
        super().__init__()
        self.lead = None
        if lead_conv:
            self.lead = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3),
                          stride=stride, padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(out_channels),
            )
        elif in_channels != out_channels:
            raise ValueError(
                "lead_conv=False requires in_channels == out_channels "
                f"(got {in_channels} and {out_channels})"
            )

        self.res = ResBlock3d(out_channels, bottleneck_ratio, dropout, negative_slope)
        self.se = SEBlock3d(out_channels, se_reduction)

    def forward(self, x: Tensor) -> Tensor:
        if self.lead is not None:
            x = self.lead(x)
        return self.se(self.res(x))


class SEResNet3D(nn.Module):
    r'''
    3D-SERESNet (Meng et al., iScience 2025): channel-stacked STFT volumes
    processed by 3D squeeze-and-excitation residual modules.

    Reconstructed from the paper's STAR Methods; see the module docstring for
    inferred choices. Pair it with the ``focal`` loss to match the paper.

    .. code-block:: yaml

        model:
          name: seresnet3d
          num_classes: 1
          kwargs:
            nperseg: 64
            n_fft: 256
            stage_channels: [16, 32, 64, 64]

    .. code-block:: python

        model = SEResNet3D(num_classes=1, in_channels=18, chunk_size=640)
        logits = model(torch.randn(2, 18, 640))  # (2, 1)

    Args:
        num_classes (int): output logits; use :obj:`1` for a single binary logit. (default: :obj:`2`)
        in_channels (int): number of EEG channels, i.e. the depth of the volume. (default: :obj:`18`)
        chunk_size (int): window length ``T`` in samples, used for the shape probe. (default: :obj:`640`)
        nperseg (int): STFT segment length in samples. (default: :obj:`64`)
        n_fft (int): FFT size; sets the number of frequency bins to ``n_fft // 2 + 1``. (default: :obj:`256`)
        overlap (float): fractional STFT overlap. (default: :obj:`0.5`)
        log_power (bool): return ``log1p`` of the power spectrogram. (default: :obj:`False`)
        normalize_input (bool): z-score each channel before the STFT, per Eq. 3. (default: :obj:`True`)
        stage_channels (tuple): stem width followed by one width per SE-ResNet
            module. (default: :obj:`(32, 32, 64, 64)`)
        head_channels (int): channels of the head convolution. (default: :obj:`128`)
        fc_hidden (int): width of the first fully connected layer. (default: :obj:`64`)
        bottleneck_ratio (float): residual bottleneck width ratio. (default: :obj:`0.5`)
        se_reduction (int): SE bottleneck ratio. (default: :obj:`16`)
        dropout (float): dropout in residual blocks and before the classifier. (default: :obj:`0.1`)
        negative_slope (float): LeakyReLU slope. (default: :obj:`0.01`)
    '''

    def __init__(self,
                 num_classes: int = 2,
                 in_channels: int = 18,
                 chunk_size: int = 640,
                 nperseg: int = 64,
                 n_fft: int = 256,
                 overlap: float = 0.5,
                 log_power: bool = False,
                 normalize_input: bool = True,
                 stage_channels: Tuple[int, ...] = (32, 32, 64, 64),
                 head_channels: int = 128,
                 fc_hidden: int = 64,
                 bottleneck_ratio: float = 0.5,
                 se_reduction: int = 16,
                 dropout: float = 0.1,
                 negative_slope: float = 0.01):
        super().__init__()
        if len(stage_channels) < 2:
            raise ValueError("stage_channels needs a stem width plus at least one module width")
        if nperseg > chunk_size:
            raise ValueError(f"nperseg ({nperseg}) exceeds the window length ({chunk_size})")
        if n_fft < nperseg:
            raise ValueError(f"n_fft ({n_fft}) must be >= nperseg ({nperseg})")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.chunk_size = chunk_size
        self.nperseg = nperseg
        self.n_fft = n_fft
        self.hop = max(1, int(round(nperseg * (1.0 - overlap))))
        self.log_power = log_power
        self.normalize_input = normalize_input
        self.negative_slope = negative_slope
        self.register_buffer("stft_win", torch.hann_window(nperseg), persistent=False)

        stem_channels, *module_channels = stage_channels

        # Stem, Eq. 4: the only convolution that mixes electrodes.
        self.stem_conv = nn.Conv3d(1, stem_channels, kernel_size=3, padding=1, bias=False)
        self.stem_pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

        modules = []
        prev = stem_channels
        for idx, channels in enumerate(module_channels):
            modules.append(
                SEResNetModule(
                    prev, channels,
                    # Module 1 acts directly on U (Eq. 5); later modules get the
                    # leading conv layer of Eq. 6, which also downsamples.
                    stride=(1, 2, 2) if idx > 0 else (1, 1, 1),
                    lead_conv=(idx > 0) or (prev != channels),
                    bottleneck_ratio=bottleneck_ratio,
                    dropout=dropout,
                    se_reduction=se_reduction,
                    negative_slope=negative_slope,
                )
            )
            prev = channels
        self.modules_3d = nn.Sequential(*modules)

        # Head, Eq. 7.
        self.head_conv = nn.Sequential(
            nn.Conv3d(prev, head_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(head_channels),
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(head_channels, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)

    def volume(self, x: Tensor) -> Tensor:
        """Z-score, STFT, and stack per channel into ``(B, 1, C, F, frames)``."""
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T) input, got shape {tuple(x.shape)}")
        b, c, t = x.shape

        if self.normalize_input:  # Eq. 3
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
            x = (x - mean) / std

        spec = torch.stft(
            x.reshape(b * c, t),
            n_fft=self.n_fft,
            hop_length=self.hop,
            win_length=self.nperseg,
            window=self.stft_win.to(x.dtype),
            center=False,
            return_complex=True,
        )
        power = spec.real.pow(2) + spec.imag.pow(2)
        if self.log_power:
            power = torch.log1p(power)
        return power.reshape(b, 1, c, power.shape[-2], power.shape[-1])

    def features(self, x: Tensor) -> Tensor:
        """Backbone forward up to the pooled feature vector ``F1`` of Eq. 7."""
        h = self.stem_pool(F.leaky_relu(self.stem_conv(self.volume(x)), self.negative_slope))
        h = self.modules_3d(h)
        h = F.leaky_relu(self.head_conv(h), self.negative_slope)
        h = F.leaky_relu(self.pool(h).flatten(1), self.negative_slope)
        return h

    def forward(self, x: Tensor) -> Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG window, shape :obj:`(B, C, T)`.

        Returns:
            torch.Tensor: logits, shape :obj:`(B, num_classes)`.
        '''
        h = self.features(x)
        h = F.leaky_relu(self.fc1(self.dropout(h)), self.negative_slope)
        return self.fc2(h)


# ---- seizure_pred registry glue ----
from seizure_pred.core.config import ModelConfig
from seizure_pred.training.registries import MODELS


@MODELS.register("seresnet3d", help="3D-SERESNet: stacked STFT volumes + 3D SE residual modules (iScience 2025).")
def build_seresnet3d(cfg: ModelConfig) -> nn.Module:
    kw = dict(getattr(cfg, "kwargs", {}) or {})
    stage_channels = tuple(int(c) for c in kw.get("stage_channels", (32, 32, 64, 64)))
    return SEResNet3D(
        num_classes=int(getattr(cfg, "num_classes", 2)),
        in_channels=int(cfg.in_channels or kw.get("in_channels", 18)),
        chunk_size=int(kw.get("chunk_size", kw.get("seq_len", 640))),
        nperseg=int(kw.get("nperseg", 64)),
        n_fft=int(kw.get("n_fft", 256)),
        overlap=float(kw.get("overlap", 0.5)),
        log_power=bool(kw.get("log_power", False)),
        normalize_input=bool(kw.get("normalize_input", True)),
        stage_channels=stage_channels,
        head_channels=int(kw.get("head_channels", 128)),
        fc_hidden=int(kw.get("fc_hidden", 64)),
        bottleneck_ratio=float(kw.get("bottleneck_ratio", 0.5)),
        se_reduction=int(kw.get("se_reduction", 16)),
        dropout=float(kw.get("dropout", 0.1)),
        negative_slope=float(kw.get("negative_slope", 0.01)),
    )
