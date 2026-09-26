"""Regression tests for the optional input BatchNorm of EEGWaveNet.

The BatchNorm lives in ``EEGWaveletEmbeddingNet`` and is applied to the
concatenated wavelet input *before* it is split into per-band branches, so the
relative power of the bands is preserved. ``input_bn="none"`` is the default and
keeps the original architecture and ``state_dict`` keys.

Supported modes:

* ``"none"`` - no input normalization (default);
* ``"per_channel"`` - one running mean/variance per EEG channel;
* ``"global"`` - a single running mean/variance for all channels and bands.
"""
from __future__ import annotations

import pytest
import torch

from seizure_pred.core.config import ModelConfig
from seizure_pred.models.eegwavenet import EEGWaveNet

CHANNELS = 18
SAMPLES = 640
COMPONENT_EDGES = [0, 320, 480, 560, 600, 640]


def _build_from_registry(**model_kwargs):
    import seizure_pred.models as models
    from seizure_pred.training.registries import MODELS

    models.register_all()
    cfg = ModelConfig(name="eegwavenet", num_classes=1, kwargs=model_kwargs)
    return MODELS.create("eegwavenet", cfg)


def test_input_bn_defaults_to_none_and_keeps_state_dict_unchanged():
    model = EEGWaveNet(n_classes=1, model_size="tiny")
    assert model.embedding.input_bn_mode == "none"
    assert model.embedding.input_bn is None
    assert not any("input_bn" in key for key in model.state_dict())


@pytest.mark.parametrize(
    ("mode", "num_features"),
    [("per_channel", CHANNELS), ("global", 1)],
)
def test_input_bn_modes_create_expected_batchnorm(mode, num_features):
    model = EEGWaveNet(n_classes=1, model_size="tiny", input_bn=mode)
    bn = model.embedding.input_bn
    assert model.embedding.input_bn_mode == mode
    assert isinstance(bn, torch.nn.BatchNorm1d)
    assert bn.num_features == num_features


@pytest.mark.parametrize("mode", ["per_channel", "global"])
def test_input_bn_runs_forward_backward_and_updates_running_stats(mode):
    torch.manual_seed(0)
    model = EEGWaveNet(n_classes=1, model_size="tiny", input_bn=mode)
    x = torch.randn(4, CHANNELS, SAMPLES) * 5.0 + 3.0

    model.train()
    out = model(x)
    assert out.shape == (4, 1)
    assert torch.isfinite(out).all()
    out.sum().backward()

    bn = model.embedding.input_bn
    assert bn.weight.grad is not None
    assert bn.running_mean.abs().sum() > 0

    model.eval()
    with torch.no_grad():
        assert torch.isfinite(model(x)).all()


@pytest.mark.parametrize("mode", ["per_channel", "global"])
def test_input_bn_preserves_relative_band_power(mode):
    """Every band of a channel must share the same normalization scale."""
    torch.manual_seed(0)
    model = EEGWaveNet(n_classes=1, model_size="tiny", input_bn=mode)
    bn = model.embedding.input_bn
    if mode == "per_channel":
        bn.running_mean.copy_(torch.arange(CHANNELS).float())
        bn.running_var.copy_(torch.arange(CHANNELS).float() + 1.0)
    else:
        bn.running_mean.fill_(2.0)
        bn.running_var.fill_(4.0)
    bn.eval()

    x = torch.randn(2, CHANNELS, SAMPLES) * 3.0 + 7.0
    with torch.no_grad():
        normalized = model.embedding._apply_input_bn(x)

    for channel in range(CHANNELS):
        scales = []
        for start, stop in zip(COMPONENT_EDGES[:-1], COMPONENT_EDGES[1:]):
            raw = x[:, channel, start:stop].std()
            out = normalized[:, channel, start:stop].std()
            scales.append((out / raw).item())
        assert max(scales) - min(scales) < 1e-4


@pytest.mark.parametrize("invalid", ["bogus", True, None])
def test_input_bn_rejects_invalid_modes(invalid):
    with pytest.raises(ValueError, match="input_bn"):
        EEGWaveNet(n_classes=1, model_size="tiny", input_bn=invalid)


def test_builder_reads_input_bn_kwarg():
    disabled = _build_from_registry(model_size="tiny")
    per_channel = _build_from_registry(model_size="tiny", input_bn="per_channel")
    global_bn = _build_from_registry(model_size="tiny", input_bn="global")

    assert disabled.embedding.input_bn is None
    assert per_channel.embedding.input_bn.num_features == CHANNELS
    assert global_bn.embedding.input_bn.num_features == 1
