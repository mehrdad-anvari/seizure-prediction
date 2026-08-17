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
