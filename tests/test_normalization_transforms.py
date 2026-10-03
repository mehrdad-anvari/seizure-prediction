import numpy as np
import pytest

from seizure_pred.transforms.registry import create_transform


@pytest.mark.parametrize("name", ["instance_norm", "robust_norm"])
def test_offline_normalization_is_per_segment_and_channel(name):
    transform = create_transform(name)
    segment = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 100.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ],
        dtype=np.float32,
    )

    normalized = transform(segment)

    assert normalized.shape == segment.shape
    assert np.isfinite(normalized).all()
    if name == "instance_norm":
        np.testing.assert_allclose(normalized.mean(axis=-1), 0.0, atol=1e-6)
    else:
        np.testing.assert_allclose(np.median(normalized, axis=-1), 0.0, atol=1e-6)
        mad = np.median(
            np.abs(normalized - np.median(normalized, axis=-1, keepdims=True)),
            axis=-1,
        )
        np.testing.assert_allclose(mad, 1.0 / 1.4826, rtol=1e-5)


@pytest.mark.parametrize("name", ["instance_norm", "robust_norm"])
def test_normalization_is_unit_invariant_for_volts_scale_data(name):
    rng = np.random.default_rng(0)
    # CHB-MIT is stored in volts: per-channel std ~2e-5
    x = (rng.standard_normal((4, 1280)) * 2e-5).astype(np.float64)
    transform = create_transform(name)

    out = transform(x)

    assert np.isfinite(out).all()
    if name == "instance_norm":
        np.testing.assert_allclose(out.std(axis=-1, ddof=1), 1.0, rtol=1e-3)
    else:
        mad = np.median(np.abs(out - np.median(out, axis=-1, keepdims=True)), axis=-1)
        np.testing.assert_allclose(mad * 1.4826, 1.0, rtol=1e-2)  # torch median is the lower median
        np.testing.assert_allclose(out.std(axis=-1), 1.0, rtol=0.1)

    np.testing.assert_allclose(transform(x * 1e6), out, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("name", ["instance_norm", "robust_norm"])
def test_normalization_zero_and_constant_input_stay_finite(name):
    transform = create_transform(name)
    for x in (np.zeros((3, 64)), np.full((3, 64), 3e-5)):
        out = transform(x)
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_absolute_eps_mode_reproduces_legacy_behaviour():
    x = np.random.default_rng(1).standard_normal((2, 500)) * 2e-5
    legacy = create_transform("instance_norm", eps_mode="absolute")(x)
    expected = (x - x.mean(-1, keepdims=True)) / (x.std(-1, ddof=1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(legacy, expected, rtol=1e-6)
    with pytest.raises(ValueError):
        create_transform("robust_norm", eps_mode="bogus")
