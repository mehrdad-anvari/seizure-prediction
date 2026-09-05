"""Smoke tests for the models reconstructed from the papers in ``papers/``.

These check the tensor plumbing -- registry build, logit shape for binary and
multiclass heads, gradient flow, and each model's documented input guards. They
cannot check fidelity to the papers, since none of the five publishes reference
code.
"""
from __future__ import annotations

import math

import pytest
import torch

PAPER_MODELS = ["fapex", "md_rescapsnet", "seizurenet_kan", "seresnet3d", "sbtm"]

# Small but valid configurations: enough samples for the STFT/derivative
# front-ends, small enough to stay quick on CPU.
SMALL_KWARGS = {
    "fapex": {"patch_size": 32, "d_model": 16, "d_state": 4, "depth": 1},
    "md_rescapsnet": {"sfreq": 128, "stage_channels": [8, 16], "stft_window": 64},
    "seizurenet_kan": {"hidden": 16, "grid_size": 5},
    "seresnet3d": {"stage_channels": [4, 4], "nperseg": 64, "n_fft": 64, "head_channels": 8},
    "sbtm": {"sfreq": 128, "num_steps": 4, "hidden_size": 16, "dense_hidden": 8},
}

CHANNELS = 8
SAMPLES = 256


def _set_threads():
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _build(name: str, *, num_classes: int = 1, **overrides):
    import seizure_pred.models as models
    from seizure_pred.core.config import ModelConfig
    from seizure_pred.training.registries import MODELS

    models.register_all()
    if name not in MODELS:
        pytest.skip(f"model '{name}' is not registered (missing optional dependency)")

    kwargs = dict(SMALL_KWARGS.get(name, {}))
    kwargs.setdefault("chunk_size", SAMPLES)
    kwargs.update(overrides)
    cfg = ModelConfig(
        name=name,
        num_classes=num_classes,
        in_channels=CHANNELS,
        sfreq=kwargs.get("sfreq"),
        kwargs=kwargs,
    )
    return MODELS.create(name, cfg)


@pytest.mark.parametrize("name", PAPER_MODELS)
def test_binary_logits_and_backward(name):
    _set_threads()
    torch.manual_seed(0)
    model = _build(name)
    out = model(torch.randn(2, CHANNELS, SAMPLES))
    assert out.shape == (2, 1)
    assert torch.isfinite(out).all()

    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)


@pytest.mark.parametrize("name", PAPER_MODELS)
def test_multiclass_logits(name):
    _set_threads()
    torch.manual_seed(0)
    model = _build(name, num_classes=2).eval()
    with torch.no_grad():
        assert model(torch.randn(2, CHANNELS, SAMPLES)).shape == (2, 2)


@pytest.mark.parametrize("name", PAPER_MODELS)
def test_rejects_non_3d_input(name):
    _set_threads()
    model = _build(name)
    with pytest.raises(Exception):
        model(torch.randn(2, CHANNELS))


# ---------------------------------------------------------------- FAPEX parts
def test_frft_matches_dft_at_quarter_turn():
    """theta = pi/2 must reproduce the centred DFT, the transform's fixed point."""
    from seizure_pred.models.fapex import frft

    torch.manual_seed(0)
    x = torch.randn(1, 1, 16, 1)
    out = frft(x, torch.full((1,), math.pi / 2))

    expected = torch.fft.fftshift(
        torch.fft.fft(torch.fft.ifftshift(x.to(torch.complex64), dim=-2), dim=-2), dim=-2
    )
    assert torch.allclose(out, expected, atol=1e-4)


def test_frnfo_learns_fractional_order():
    from seizure_pred.models.fapex import FrNFO

    frnfo = FrNFO(d_model=4)
    amplitude, phase = frnfo(torch.randn(2, 3, 8, 4))
    assert amplitude.shape == (2, 3, 8, 4)
    assert (amplitude >= 0).all()
    assert phase.abs().max() <= math.pi + 1e-5

    amplitude.sum().backward()
    assert frnfo.theta_raw.grad is not None
    assert (frnfo.theta > 0).all() and (frnfo.theta < math.pi).all()


# ------------------------------------------------------- MD-ResCapsNet parts
def test_csp_filters_shape_and_use():
    from seizure_pred.models.md_rescapsnet import compute_csp_filters

    torch.manual_seed(0)
    pre = torch.randn(6, CHANNELS, SAMPLES)
    inter = torch.randn(6, CHANNELS, SAMPLES) * 2.0
    filters = compute_csp_filters(pre, inter, n_components=4)
    assert filters.shape == (4, CHANNELS)

    model = _build("md_rescapsnet", csp_filters=filters.tolist())
    with torch.no_grad():
        assert model(torch.randn(2, CHANNELS, SAMPLES)).shape == (2, 1)


def test_capsule_lengths_are_bounded():
    model = _build("md_rescapsnet").eval()
    with torch.no_grad():
        lengths = model.capsule_lengths(torch.randn(2, CHANNELS, SAMPLES))
    assert ((lengths >= 0) & (lengths <= 1)).all()


def test_capsule_margin_loss_prefers_correct_capsule():
    import seizure_pred.training.components.losses  # noqa: F401  (registers losses)
    from seizure_pred.training.registries import LOSSES

    loss_fn = LOSSES.create("capsule_margin")
    targets = torch.tensor([1.0, 0.0])
    confident = torch.tensor([[4.0], [-4.0]])
    wrong = torch.tensor([[-4.0], [4.0]])
    assert loss_fn(confident, targets) < loss_fn(wrong, targets)


# ------------------------------------------------------ SeizureNet-KAN parts
def test_plv_adjacency_is_symmetric_and_thresholded():
    from seizure_pred.models.seizurenet_kan import plv_adjacency

    torch.manual_seed(0)
    x = torch.randn(2, CHANNELS, SAMPLES)
    adj = plv_adjacency(x)
    assert adj.shape == (2, CHANNELS, CHANNELS)
    assert torch.allclose(adj, adj.transpose(1, 2), atol=1e-5)
    assert (adj.diagonal(dim1=1, dim2=2) == 0).all()
    assert (adj >= 0).all() and (adj <= 1).all()
    # The median+std threshold must zero out some genuine off-diagonal edges.
    off_diag = adj[:, ~torch.eye(CHANNELS, dtype=torch.bool)]
    assert (off_diag == 0).any() and (off_diag > 0).any()


def test_kan_layer_spline_branch_is_used():
    from seizure_pred.models.seizurenet_kan import KANLinear

    layer = KANLinear(3, 2, grid_size=5, spline_order=3)
    x = torch.zeros(4, 3)  # inside the grid, so splines contribute
    bases = layer.b_splines(x)
    assert bases.shape == (4, 3, 5 + 3)
    assert bases.sum() > 0
    assert layer(x).shape == (4, 2)


def test_seizurenet_kan_decoder_reconstructs_node_features():
    model = _build("seizurenet_kan")
    x = torch.randn(2, CHANNELS, SAMPLES)
    mask = torch.zeros(2, CHANNELS, dtype=torch.bool)
    mask[:, 0] = True
    z = model.node_embeddings(x, node_mask=mask)
    assert z.shape == (2, CHANNELS, 16)
    assert model.decoder(z).shape == (2, CHANNELS, 5)


# ------------------------------------------------------------- SBTM features
def test_hjorth_and_spectral_features_are_finite_on_flat_input():
    from seizure_pred.models.sbtm import fused_features

    flat = torch.zeros(2, CHANNELS, 64)  # degenerate: zero variance everywhere
    feats = fused_features(flat, sfreq=128.0)
    assert feats.shape == (2, CHANNELS, 13)
    assert torch.isfinite(feats).all()


def test_sbtm_rejects_too_many_steps():
    with pytest.raises(ValueError):
        _build("sbtm", num_steps=SAMPLES)  # one sample per step
