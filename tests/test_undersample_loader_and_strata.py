import warnings

import numpy as np
import pytest
import torch

from seizure_pred.data.splits import split_into_strata
from seizure_pred.training.dataloaders.undersample import UnderSampledDataLoader


class _ToyDataset:
    def __init__(self, n_pos, n_neg):
        self.y = np.array([1] * n_pos + [0] * n_neg)

    def __len__(self):
        return len(self.y)

    def get_class_indices(self):
        return np.where(self.y == 1)[0], np.where(self.y == 0)[0]

    def __getitem__(self, i):
        return torch.zeros(2, 4), torch.tensor(int(self.y[i])), {}


def _sizes(loader):
    return [len(y) for _, y, _ in loader]


def test_train_loader_drops_trailing_singleton_batch():
    # 2 * 5 = 10 samples, batch 3 -> 3,3,3,1 ; trailing 1 dropped
    loader = UnderSampledDataLoader(_ToyDataset(5, 20), batch_size=3, shuffle=True)
    assert _sizes(loader) == [3, 3, 3]
    assert len(loader) == 3


def test_eval_loader_keeps_all_samples():
    loader = UnderSampledDataLoader(_ToyDataset(5, 20), batch_size=3, shuffle=False)
    assert _sizes(loader) == [3, 3, 3, 1]
    assert len(loader) == 4


def test_non_singleton_trailing_batch_kept_and_opt_out():
    loader = UnderSampledDataLoader(_ToyDataset(4, 20), batch_size=3, shuffle=True)  # 8 -> 3,3,2
    assert _sizes(loader) == [3, 3, 2] and len(loader) == 3
    loader = UnderSampledDataLoader(
        _ToyDataset(5, 20), batch_size=3, shuffle=True, drop_last_singleton=False
    )
    assert _sizes(loader) == [3, 3, 3, 1] and len(loader) == 4


def test_split_into_strata_clamps_m_with_warning():
    idx = np.arange(34)
    with pytest.warns(UserWarning, match="length=34.*clamping M to the maximum valid value 6"):
        clamped = split_into_strata(idx, N=5, M=10)
    expected = split_into_strata(idx, N=5, M=6)
    assert len(clamped) == len(expected)
    for a, b in zip(clamped, expected):
        np.testing.assert_array_equal(a, b)


def test_split_into_strata_valid_m_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        split_into_strata(np.arange(34), N=5, M=6)
