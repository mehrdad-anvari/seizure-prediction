from __future__ import annotations

import numpy as np
import pytest

from seizure_pred.data.chbmit_npz import CHBMITDataset

SUFFIX = "_fd_5s_prex5"


def _write_session(root, ses, event_ids, n_per=2, C=2, T=8):
    d = root / "sub-01" / ses / "eeg"
    d.mkdir(parents=True)
    labels, ev = [], []
    for e in event_ids:
        labels += [e.rsplit("_", 1)[0]] * n_per
        ev += [e] * n_per
    n = len(ev)
    meta = {
        "event_id": ev,
        "label": labels,
        "augmented": [False] * n,
    }
    np.savez(
        d / f"sub-01_{ses}_processed{SUFFIX}_float.npz",
        X=np.zeros((n, C, T), dtype=np.float32),
        y=np.array(labels),
        meta_df=np.array(meta, dtype=object),
    )


@pytest.fixture
def bids(tmp_path):
    _write_session(tmp_path, "ses-01", ["interictal_1", "preictal_1"])
    _write_session(tmp_path, "ses-02", ["interictal_2", "preictal_2"])
    return tmp_path


def _load(root, **kw):
    return CHBMITDataset(
        dataset_dir=str(root), subject_id="01", suffix=SUFFIX, print_events=False, **kw
    )


def test_sessions_default_loads_all(bids):
    ds = _load(bids)
    assert len(ds) == 8
    assert set(ds.group_ids.tolist()) == {
        "interictal_1_0", "preictal_1_0", "interictal_2_1", "preictal_2_1"
    }


@pytest.mark.parametrize("spec", [["ses-01"], ["01"], "ses-01"])
def test_sessions_filter_keeps_original_numbering(bids, spec):
    ds = _load(bids, sessions=spec)
    assert len(ds) == 4
    assert set(ds.group_ids.tolist()) == {"interictal_1_0", "preictal_1_0"}
    ds2 = _load(bids, sessions=["02"])
    assert set(ds2.group_ids.tolist()) == {"interictal_2_1", "preictal_2_1"}


def test_sessions_missing_raises(bids):
    with pytest.raises(ValueError, match="ses-03"):
        _load(bids, sessions=["ses-03"])
