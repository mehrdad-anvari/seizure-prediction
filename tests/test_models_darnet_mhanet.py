"""Smoke tests for the DARNet and MHANet ports: registry build + forward/backward."""
from __future__ import annotations

import pytest
import torch


def _set_threads():
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _build(name: str, *, in_channels: int = 18, chunk_size: int = 128,
           num_classes: int = 1, **kwargs):
    import seizure_pred.models as models
    from seizure_pred.core.config import ModelConfig
    from seizure_pred.training.registries import MODELS

    models.register_all()
    if name not in MODELS:
        pytest.skip(f"model '{name}' is not registered (missing optional dependency)")
    cfg = ModelConfig(
        name=name,
        num_classes=num_classes,
        in_channels=in_channels,
        kwargs={"chunk_size": chunk_size, **kwargs},
    )
    return MODELS.create(name, cfg)


@pytest.mark.parametrize("name", ["darnet", "mhanet"])
def test_binary_logits_and_backward(name):
    _set_threads()
    torch.manual_seed(0)
    model = _build(name)
    out = model(torch.randn(2, 18, 128))
    assert out.shape == (2, 1)
    out.sum().backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("name", ["darnet", "mhanet"])
def test_multiclass_logits(name):
    _set_threads()
    torch.manual_seed(0)
    model = _build(name, num_classes=2).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 18, 128)).shape == (2, 2)


def test_mhanet_head_count_divides_channel_count():
    _set_threads()
    model = _build("mhanet", in_channels=18)
    assert 18 % model.num_heads == 0


def test_mhanet_rejects_mismatched_window_length():
    _set_threads()
    model = _build("mhanet", chunk_size=128)
    with pytest.raises(ValueError):
        model(torch.randn(2, 18, 64))


def test_darnet_requires_head_divisible_embedding():
    _set_threads()
    from seizure_pred.models.darnet import DARNet

    with pytest.raises(ValueError):
        DARNet(num_electrodes=18, chunk_size=128, d_model=10, num_heads=4)
