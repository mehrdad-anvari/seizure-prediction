from __future__ import annotations

r"""SBTM: feature-driven bidirectional LSTM for seizure prediction.

- Paper: Kumar A, Tripathi E, Tripathi A K, Diwedi H K, Rathore P S, Ansari A S.
  SBTM: epileptic seizure prediction from EEG signal using deep learning in
  blockchain-enabled smart healthcare monitoring with IoT networking.
  Scientific Reports 16:6830 (2026).
- Related project: none published.

Reconstruction from the paper text. What this file covers, and what it does not:

The paper's architecture is deliberately simple -- "the input layer, a BiLSTM
layer, dense layers, and a softmax layer" over a fused feature vector. Its stated
contribution is elsewhere: the **Spizella optimizer**, a bio-inspired metaheuristic
that replaces Adam for weight tuning, plus a blockchain/IoT deployment wrapper.
Neither is implemented here. The metaheuristic is a training procedure, not part
of the network, and it is incompatible with this library's gradient-based
Trainer; the blockchain and IoT layers are infrastructure, outside the scope of a
model zoo. Train this with the registry's ``adam``/``adamw``/``sgd`` instead, and
expect that to differ from the paper's optimisation.

Implemented, and stated by the paper:

1. **Feature extraction.** Spectral, Hjorth, and statistical features per channel,
   concatenated to a fused vector -- 13 features per channel, matching the paper's
   stated ``(N, 13)`` fused dimension: 5 statistical (mean, variance, skewness,
   kurtosis, peak-to-peak), 3 Hjorth (activity, mobility, complexity), and 5
   spectral (centroid, spread, kurtosis, skewness, crest).
2. **Bi-LSTM** over the feature sequence, reading forward and backward.
3. **Dense head** producing class scores.

Inferred, because the paper does not state them:

- Features are computed per sub-window rather than once per record, which is what
  gives the recurrence something sequential to read. ``num_steps`` sets how many.
- Hidden width (128), LSTM depth (1) and dense width (64). The paper leaves these
  to the Spizella search, so no published values exist.
- Feature normalisation before the recurrence. A variance and a spectral centroid
  in Hz differ by orders of magnitude, which an LSTM handles poorly unstandardised.
- Sequence pooling concatenates the last forward and backward states, the usual
  reading of "both the forward and backward flow of information".

Input is ``(B, C, T)`` raw EEG.
"""

import torch
import torch.nn as nn

Tensor = torch.Tensor

NUM_FEATURES_PER_CHANNEL = 13


def statistical_features(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Mean, variance, skewness, kurtosis, peak-to-peak; ``(..., T)`` -> ``(..., 5)``."""
    mean = x.mean(dim=-1)
    centred = x - mean.unsqueeze(-1)
    var = centred.pow(2).mean(dim=-1)
    std = (var + eps).sqrt()
    skew = centred.pow(3).mean(dim=-1) / std.pow(3).clamp_min(eps)
    kurt = centred.pow(4).mean(dim=-1) / var.pow(2).clamp_min(eps)
    p2p = x.amax(dim=-1) - x.amin(dim=-1)
    return torch.stack([mean, var, skew, kurt, p2p], dim=-1)


def hjorth_features(x: Tensor, eps: float = 1e-8) -> Tensor:
    r"""Hjorth activity, mobility and complexity (Table 2); ``(..., T)`` -> ``(..., 3)``.

    Activity is the signal variance, mobility the variance ratio of the first
    derivative to the signal, and complexity the same ratio one derivative up,
    divided by mobility.
    """
    d1 = x[..., 1:] - x[..., :-1]
    d2 = d1[..., 1:] - d1[..., :-1]

    var0 = x.var(dim=-1, unbiased=False)
    var1 = d1.var(dim=-1, unbiased=False)
    var2 = d2.var(dim=-1, unbiased=False)

    activity = var0
    mobility = (var1 / var0.clamp_min(eps)).clamp_min(0.0).sqrt()
    complexity = (var2 / var1.clamp_min(eps)).clamp_min(0.0).sqrt() / mobility.clamp_min(eps)
    return torch.stack([activity, mobility, complexity], dim=-1)


def spectral_features(x: Tensor, sfreq: float, eps: float = 1e-8) -> Tensor:
    """Spectral centroid, spread, kurtosis, skewness, crest (Table 3); -> ``(..., 5)``."""
    n = x.shape[-1]
    spectrum = torch.fft.rfft(x, dim=-1)
    power = spectrum.real.pow(2) + spectrum.imag.pow(2)
    freqs = torch.fft.rfftfreq(n, d=1.0 / sfreq, device=x.device, dtype=x.dtype)

    total = power.sum(dim=-1).clamp_min(eps)
    centroid = (power * freqs).sum(dim=-1) / total
    deviation = freqs - centroid.unsqueeze(-1)
    spread = ((power * deviation.pow(2)).sum(dim=-1) / total).clamp_min(0.0).sqrt()

    spread_safe = spread.clamp_min(eps)
    kurtosis = (power * deviation.pow(4)).sum(dim=-1) / (total * spread_safe.pow(4))
    skewness = (power * deviation.pow(3)).sum(dim=-1) / (total * spread_safe.pow(3))
    crest = power.amax(dim=-1) / power.mean(dim=-1).clamp_min(eps)

    return torch.stack([centroid, spread, kurtosis, skewness, crest], dim=-1)


def fused_features(x: Tensor, sfreq: float) -> Tensor:
    """Concatenate the three feature families (Eq. 2); ``(..., T)`` -> ``(..., 13)``."""
    return torch.cat(
        [statistical_features(x), hjorth_features(x), spectral_features(x, sfreq)],
        dim=-1,
    )


class SBTM(nn.Module):
    r'''
    SBTM (Kumar et al., Scientific Reports 2026): spectral + Hjorth + statistical
    features per sub-window, read by a bidirectional LSTM and a dense head.

    The paper's Spizella metaheuristic optimiser and its blockchain/IoT wrapper are
    out of scope -- see the module docstring. Train with a standard optimiser.

    .. code-block:: yaml

        model:
          name: sbtm
          num_classes: 1
          sfreq: 128
          kwargs:
            num_steps: 10
            hidden_size: 128

    .. code-block:: python

        model = SBTM(num_classes=1, in_channels=18, chunk_size=640, sfreq=128)
        logits = model(torch.randn(4, 18, 640))  # (4, 1)

    Args:
        num_classes (int): output logits; use :obj:`1` for a single binary logit. (default: :obj:`2`)
        in_channels (int): number of EEG channels. (default: :obj:`18`)
        chunk_size (int): window length ``T`` in samples. (default: :obj:`640`)
        sfreq (float): sampling rate in Hz, used by the spectral features. (default: :obj:`128.0`)
        num_steps (int): sub-windows the window is split into, i.e. the recurrence
            length. (default: :obj:`10`)
        hidden_size (int): LSTM hidden width per direction. (default: :obj:`128`)
        num_layers (int): stacked LSTM layers. (default: :obj:`1`)
        dense_hidden (int): width of the dense layer before the classifier. (default: :obj:`64`)
        dropout (float): dropout before the classifier (and between LSTM layers
            when ``num_layers > 1``). (default: :obj:`0.2`)
    '''

    def __init__(self,
                 num_classes: int = 2,
                 in_channels: int = 18,
                 chunk_size: int = 640,
                 sfreq: float = 128.0,
                 num_steps: int = 10,
                 hidden_size: int = 128,
                 num_layers: int = 1,
                 dense_hidden: int = 64,
                 dropout: float = 0.2):
        super().__init__()
        if num_steps < 1:
            raise ValueError(f"num_steps must be >= 1, got {num_steps}")
        step_len = chunk_size // num_steps
        if step_len < 8:
            raise ValueError(
                f"num_steps={num_steps} leaves only {step_len} samples per step; "
                "the derivative and spectral features need at least 8"
            )

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.chunk_size = chunk_size
        self.sfreq = float(sfreq)
        self.num_steps = num_steps
        self.step_len = step_len

        feature_dim = in_channels * NUM_FEATURES_PER_CHANNEL
        self.feature_norm = nn.BatchNorm1d(feature_dim)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(2 * hidden_size, dense_hidden)
        self.classifier = nn.Linear(dense_hidden, num_classes)

    def feature_sequence(self, x: Tensor) -> Tensor:
        """Split into sub-windows and extract features; ``(B, C, T)`` -> ``(B, steps, C*13)``."""
        if x.dim() != 3:
            raise ValueError(f"expected (B, C, T) input, got shape {tuple(x.shape)}")
        b, c, t = x.shape
        usable = self.num_steps * self.step_len
        if t < usable:
            raise ValueError(
                f"window length {t} is shorter than num_steps * step_len ({usable})"
            )

        steps = x[..., :usable].reshape(b, c, self.num_steps, self.step_len)
        feats = fused_features(steps, self.sfreq)  # (B, C, steps, 13)
        return feats.permute(0, 2, 1, 3).reshape(b, self.num_steps, -1)

    def forward(self, x: Tensor) -> Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG window, shape :obj:`(B, C, T)`.

        Returns:
            torch.Tensor: logits, shape :obj:`(B, num_classes)`.
        '''
        seq = self.feature_sequence(x)
        seq = self.feature_norm(seq.transpose(1, 2)).transpose(1, 2)

        out, _ = self.lstm(seq)
        # Last forward step and last backward step, i.e. both directions' final view.
        half = out.shape[-1] // 2
        pooled = torch.cat([out[:, -1, :half], out[:, 0, half:]], dim=-1)

        h = torch.relu(self.dense(self.dropout(pooled)))
        return self.classifier(h)


# ---- seizure_pred registry glue ----
from seizure_pred.core.config import ModelConfig
from seizure_pred.training.registries import MODELS


@MODELS.register("sbtm", help="SBTM: spectral/Hjorth/statistical features + Bi-LSTM (Sci Reports 2026).")
def build_sbtm(cfg: ModelConfig) -> nn.Module:
    kw = dict(getattr(cfg, "kwargs", {}) or {})
    sfreq = float(getattr(cfg, "sfreq", None) or kw.get("sfreq", 128.0))
    return SBTM(
        num_classes=int(getattr(cfg, "num_classes", 2)),
        in_channels=int(cfg.in_channels or kw.get("in_channels", 18)),
        chunk_size=int(kw.get("chunk_size", kw.get("seq_len", 640))),
        sfreq=sfreq,
        num_steps=int(kw.get("num_steps", 10)),
        hidden_size=int(kw.get("hidden_size", 128)),
        num_layers=int(kw.get("num_layers", 1)),
        dense_hidden=int(kw.get("dense_hidden", 64)),
        dropout=float(kw.get("dropout", 0.2)),
    )
