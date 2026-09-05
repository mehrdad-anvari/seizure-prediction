from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


class MultiscaleTemporalLayer(nn.Module):
    """Multi-scale temporal convolution layer.

    Args:
        seq_len (int): The sequence length.
        kernel_size (int): The kernel size for convolution.
    """

    def __init__(self, seq_len: int, kernel_size: int):
        super(MultiscaleTemporalLayer, self).__init__()

        self.multiscale_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding='same'
        )
        self.act = nn.ELU()
        self.norm = nn.LayerNorm(seq_len)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.multiscale_conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class MultiscaleTemporalAttention(nn.Module):
    """Multi-scale temporal attention module.

    Args:
        num_electrodes (int): The number of EEG electrodes.
        chunk_size (int): The number of data points in each EEG chunk.
    """

    def __init__(self, num_electrodes: int, chunk_size: int):
        super(MultiscaleTemporalAttention, self).__init__()

        self.spatio_conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(num_electrodes, 1)
        )
        self.up_channel_conv = nn.Conv1d(
            in_channels=1,
            out_channels=3,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.project_out = nn.Conv2d(
            in_channels=1,
            out_channels=num_electrodes,
            kernel_size=1,
            stride=1
        )

        self.multi_temporal_k_2 = MultiscaleTemporalLayer(
            chunk_size, kernel_size=2)
        self.multi_temporal_k_4 = MultiscaleTemporalLayer(
            chunk_size, kernel_size=4)
        self.multi_temporal_k_6 = MultiscaleTemporalLayer(
            chunk_size, kernel_size=6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.permute(0, 2, 1, 3)
        x = self.spatio_conv(x)
        x = self.up_channel_conv(x.squeeze(2))

        x, y, z = x.chunk(3, dim=1)

        x_attn = self.multi_temporal_k_2(x)
        y_attn = self.multi_temporal_k_4(y)
        z_attn = self.multi_temporal_k_6(z)

        out = x_attn * x + y_attn * y + z_attn * z
        out = out.view(batch_size, 1, 1, -1)
        out = self.project_out(out)
        return out


class ChannelAttention(nn.Module):
    """Channel attention module with multi-scale temporal attention.

    Args:
        num_electrodes (int): The number of EEG electrodes.
        chunk_size (int): The number of data points in each EEG chunk.
        num_heads (int): The number of attention heads. Must divide
            :obj:`num_electrodes`.
        bias (bool): Whether to use bias in convolution layers.
    """

    def __init__(self,
                 num_electrodes: int,
                 chunk_size: int,
                 num_heads: int,
                 bias: bool = False):
        super(ChannelAttention, self).__init__()

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(
            num_electrodes, num_electrodes * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            num_electrodes * 3,
            num_electrodes * 3,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=num_electrodes * 3,
            bias=bias
        )
        self.project_out = nn.Conv2d(
            num_electrodes, num_electrodes, kernel_size=1, bias=bias)

        self.multiscale_temporal_attention = MultiscaleTemporalAttention(
            num_electrodes,
            chunk_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        v = self.multiscale_temporal_attention(v)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out


class MultiscaleGlobalAttention(nn.Module):
    """Multi-scale global attention module with dilated convolutions."""

    def __init__(self):
        super(MultiscaleGlobalAttention, self).__init__()

        self.down_channel = nn.Conv2d(3, 1, 1, 1, 0)
        self.norm = nn.BatchNorm2d(1)
        self.dilation_rate = 3

        self.conv_0 = nn.Conv2d(1, 1, 3, padding='same', dilation=1)
        self.conv_1 = nn.Conv2d(1, 1, 5, padding='same', dilation=2)
        self.conv_2 = nn.Conv2d(1, 1, 7, padding='same',
                                dilation=self.dilation_rate)

        self.up_channel = nn.Sequential(
            nn.Conv2d(1, 3, 1, 1, 0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x.clone()
        x = self.norm(x)
        x = self.up_channel(x)
        y = x.clone()

        y1, y2, y3 = torch.chunk(y, 3, dim=1)

        attn_0 = self.conv_0(y1) * y1
        attn_1 = self.conv_1(y2) * y2
        attn_2 = self.conv_2(y3) * y3

        attn = torch.cat([attn_0, attn_1, attn_2], dim=1)
        out = x * attn
        out = self.down_channel(out) + shortcut

        return out


class SpatiotemporalConvolution(nn.Module):
    """Spatiotemporal convolution module.

    Args:
        num_electrodes (int): The number of EEG electrodes.
        out_channels (int): The number of temporal/spatial filters.
    """

    def __init__(self, num_electrodes: int, out_channels: int = 5):
        super(SpatiotemporalConvolution, self).__init__()

        self.temporal_convolution = nn.Sequential(
            nn.Conv2d(1, out_channels, (1, 2), stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )

        self.spatio_convolution = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, (num_electrodes, 1), stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ELU()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_convolution(x)
        x = self.spatio_convolution(x)
        x = self.pool(x)
        return x


class MHANet(nn.Module):
    r'''
    The MHANet model is based on the paper "MHANet: Multi-scale Hybrid Attention Network for Auditory Attention Detection". For more details, please refer to the following information.

    - Paper: Li L, Fan C, Zhang H, et al. MHANet: Multi-scale Hybrid Attention Network for Auditory Attention Detection[J]. International Joint Conference on Artificial Intelligence, 2025.
    - URL: https://arxiv.org/abs/2505.15364
    - Related Project: https://github.com/fchest/MHANet

    The model consumes raw EEG windows shaped :obj:`(B, C, T)`, which is the
    tensor layout produced by this library's dataloaders, so it can be used
    directly from a study config:

    .. code-block:: yaml

        model:
          name: mhanet
          num_classes: 1
          kwargs:
            chunk_size: 640   # must equal the window length T
            num_heads: 9      # must divide the channel count

    .. code-block:: python

        from seizure_pred.core.config import ModelConfig
        from seizure_pred.training.registries import MODELS
        import seizure_pred.models as models

        models.register_all()
        cfg = ModelConfig(name="mhanet", num_classes=1, in_channels=18,
                          kwargs={"chunk_size": 640})
        model = MODELS.create("mhanet", cfg)
        logits = model(torch.randn(4, 18, 640))  # (4, 1)

    .. note::
        The multi-scale temporal branch layer-normalizes over the time axis, so
        :obj:`chunk_size` must match the window length exactly, and
        :obj:`num_electrodes` must be divisible by :obj:`num_heads` (with 18
        bipolar CHB-MIT channels, use 1, 2, 3, 6, 9 or 18 heads).

    Args:
        num_electrodes (int): The number of electrodes. (default: :obj:`64`)
        chunk_size (int): Number of data points in each EEG chunk, i.e. :math:`T`. (default: :obj:`64`)
        num_heads (int): The number of attention heads. Must divide :obj:`num_electrodes`. (default: :obj:`16`)
        bias (bool): Whether to use bias in convolution layers. (default: :obj:`False`)
        num_classes (int): The number of output logits. Use :obj:`1` for a single binary logit. (default: :obj:`2`)
    '''

    def __init__(self,
                 num_electrodes: int = 64,
                 chunk_size: int = 64,
                 num_heads: int = 16,
                 bias: bool = False,
                 num_classes: int = 2):
        super(MHANet, self).__init__()

        if num_electrodes % num_heads != 0:
            raise ValueError(
                f"num_electrodes ({num_electrodes}) must be divisible by num_heads "
                f"({num_heads}); pick a divisor of {num_electrodes}."
            )

        self.num_electrodes = num_electrodes
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.bias = bias
        self.num_classes = num_classes

        self.channel_attention = ChannelAttention(
            num_electrodes=num_electrodes,
            chunk_size=chunk_size,
            num_heads=num_heads,
            bias=bias
        )

        self.multiscale_global_attention = MultiscaleGlobalAttention()
        self.spatiotemporal_convolution = SpatiotemporalConvolution(
            num_electrodes
        )

        self.flatten = nn.Flatten()
        self.out = nn.Linear(self.feature_dim(), num_classes)

    def feature_dim(self) -> int:
        # Run the shape probe in eval mode so the batch-norm running statistics
        # are not updated with the zero-filled mock input.
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)

                x = mock_eeg.permute(0, 2, 1, 3)
                x = self.channel_attention(x)
                x = x.permute(0, 2, 1, 3)
                x = self.multiscale_global_attention(x)
                x = self.spatiotemporal_convolution(x)
                x = self.flatten(x)

                return x.shape[1]
        finally:
            self.train(was_training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r'''
        Args:
            x (torch.Tensor): EEG signal representation, the ideal input shape is :obj:`[n, 64, 64]`. Here, :obj:`n` corresponds to the batch size, the first :obj:`64` corresponds to :obj:`num_electrodes`, and the second :obj:`64` corresponds to :obj:`chunk_size`.

        Returns:
            torch.Tensor[size of batch, number of classes]: The logits of the samples.
        '''
        if x.shape[-1] != self.chunk_size:
            raise ValueError(
                f"MHANet was built for chunk_size={self.chunk_size} but got an input "
                f"of length {x.shape[-1]}; set model.kwargs.chunk_size to the window length."
            )
        x = x.unsqueeze(1)
        x = x.permute(0, 2, 1, 3)
        x = self.channel_attention(x)
        x = x.permute(0, 2, 1, 3)
        x = self.multiscale_global_attention(x)
        x = self.spatiotemporal_convolution(x)
        x = self.flatten(x)
        x = self.out(x)
        return x


# ---- seizure_pred registry glue ----
from seizure_pred.core.config import ModelConfig
from seizure_pred.training.registries import MODELS


def _largest_divisor_at_most(n: int, cap: int) -> int:
    """Largest divisor of ``n`` that is <= ``cap`` (used to pick a head count)."""
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


@MODELS.register("mhanet", help="MHANet: multi-scale hybrid attention network (channel + global attention).")
def build_mhanet(cfg: ModelConfig) -> nn.Module:
    kw = dict(getattr(cfg, "kwargs", {}) or {})
    num_electrodes = int(cfg.in_channels or kw.get("num_electrodes", 64))
    chunk_size = int(kw.get("chunk_size", kw.get("seq_len", 640)))
    # The paper uses 16 heads on 64 electrodes; fall back to the largest head
    # count that divides the channel count of this dataset (18 -> 9).
    num_heads = int(kw.get("num_heads", _largest_divisor_at_most(num_electrodes, 16)))
    return MHANet(
        num_electrodes=num_electrodes,
        chunk_size=chunk_size,
        num_heads=num_heads,
        bias=bool(kw.get("bias", False)),
        num_classes=int(getattr(cfg, "num_classes", 2)),
    )
