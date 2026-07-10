from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BandFeatureExtractor(nn.Module):
    def __init__(self, n_channels, kernel_t_len, kernel_c_len=None, n_output_features=16, dilation=1, pool_kernel=8, pool_stride=8):
        super().__init__()
        self.n_channels = n_channels
        if kernel_c_len is None:
            kernel_c_len = n_channels  # capture global channel patterns in first layer

        # Temporal convolution: (1, kernel_t_len)
        self.temp_conv = nn.Conv2d(
            1, 16, kernel_size=(1, kernel_t_len), padding=(0, kernel_t_len//2), dilation=(1, dilation), bias=False
        )
        # Channel convolution: (kernel_c_len, 1)
        self.channel_conv = nn.Conv2d(
            16, n_output_features, kernel_size=(kernel_c_len, 1), padding=(0, 0), bias=False
        )

        self.bn = nn.BatchNorm2d(n_output_features)
        self.pool = nn.MaxPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride))
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        x: (B, channels, n_times)
        """
        B, C, T = x.shape
        # add a dummy channel dimension for Conv2d
        x = x.unsqueeze(1)  # (B, 1, C, T)

        # Temporal conv
        x = self.temp_conv(x)  # (B, 1, C, T)
        # Channel conv
        x = self.channel_conv(x)  # (B, n_output_features, H_out=1, T_out)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)  # reduce temporal dimension

        # Remove singleton channel dimension along height
        x = x.squeeze(2)  # (B, n_output_features, T_out)
        return x

class EEGBandEmbeddingNet(nn.Module):
    def __init__(self, n_channels=18, n_bands=5, kernel_t_lens=None, kernel_c_lens=None, n_output_features=16):
        super().__init__()
        if kernel_t_lens is None:
            kernel_t_lens = [7]*n_bands  # default temporal kernel lengths
        if kernel_c_lens is None:
            kernel_c_lens = [n_channels]*n_bands  # default: global channel conv first

        assert len(kernel_t_lens) == n_bands
        assert len(kernel_c_lens) == n_bands

        self.extractors = nn.ModuleList([
            BandFeatureExtractor(n_channels, kernel_t_len=kernel_t_lens[i],
                                 kernel_c_len=kernel_c_lens[i],
                                 n_output_features=n_output_features)
            for i in range(n_bands)
        ])
        self.band_conv = nn.Conv2d(16,32,kernel_size=(1,n_bands))

    def forward(self, x):
        """
        x: (B, channels, bands, n_times)
        """
        B, C, H, T = x.shape
        band_features = []
        for i in range(H):
            band_x = x[:, :, i, :]  # (B, channels, n_times)
            feat = self.extractors[i](band_x)  # (B, n_output_features, T_out)
            band_features.append(feat.unsqueeze(-1))
        # concatenate all bands
        embedding = torch.cat(band_features, dim=-1)  # (B, n_output_features * n_bands)
        embedding = self.band_conv(embedding)
        embedding = torch.flatten(embedding.mean(dim=-2), start_dim=1)
        return embedding

class EEGBandClassifier(nn.Module):
    def __init__(self, n_classes: int = 2, n_bands: int = 5, n_channels: int = 18, **kwargs):
        super().__init__()
        actual_classes = kwargs.get("num_classes", n_classes)
        actual_channels = kwargs.get("in_channels", n_channels)
        self.embedding_net = EEGBandEmbeddingNet(
            n_channels=actual_channels,
            n_bands=n_bands,
            n_output_features=16,
        )
        self.classifier = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.BatchNorm1d(32),
            nn.Linear(32, actual_classes)
        )

    def forward(self, x):
        emb = self.embedding_net(x)  # (B, features)
        logits = self.classifier(emb)
        return logits

if __name__ == "__main__":
    from network_debugger import NetworkDebugger
    from torchinfo import summary

    torch.manual_seed(1)
    model = EEGBandClassifier(n_classes=2).cuda()

    debugger = NetworkDebugger(model)
    debugger.register_hooks()

    # Fake EEG input: (B, C, Bn, T)
    x = torch.randn(2, 18, 5, 640).cuda()
    y = model(x)

    print("\nOutput shape:", y.shape)

    # Backward pass
    target = torch.randint(0, 2, (2,)).cuda()
    loss = F.cross_entropy(y, target)
    loss.backward()

    debugger.remove_hooks()

    summary(model, (2, 18, 5, 640))

from seizure_pred.core.config import ModelConfig
from seizure_pred.training.registries import MODELS

def _build_eeg_band_classifier_common(cfg: ModelConfig) -> EEGBandClassifier:
    kw = dict(getattr(cfg, "kwargs", {}) or {})
    in_ch = cfg.in_channels or kw.get("in_channels", kw.get("num_electrodes", 18))
    n_classes = cfg.num_classes or kw.get("num_classes", kw.get("n_classes", 2))
    n_bands = kw.get("n_bands", 5)
    return EEGBandClassifier(
        n_classes=int(n_classes),
        n_bands=int(n_bands),
        n_channels=int(in_ch),
        **{k: v for k, v in kw.items() if k not in {"in_channels", "num_electrodes", "num_classes", "n_classes", "n_bands"}}
    )

@MODELS.register("eeg_band_classifier", help="EEG bandpower classifier baseline.")
def build_eeg_band_classifier(cfg: ModelConfig):
    return _build_eeg_band_classifier_common(cfg)

@MODELS.register("eegbandclassifier", help="Legacy name for EEG bandpower classifier.")
def build_eegbandclassifier(cfg: ModelConfig):
    return _build_eeg_band_classifier_common(cfg)
