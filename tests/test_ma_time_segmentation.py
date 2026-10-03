from __future__ import annotations

import os
import pickle
import tempfile

import numpy as np

from seizure_pred.analysis.calibration_sweep import analyze_nested_calibration
from seizure_pred.analysis.metrics import (
    moving_average_by_segments,
    moving_average_segmented,
    segment_ids_from_meta,
)


def _meta(n, ev="interictal_1", t0=0.0, step=5.0):
    return [{"event_id": ev, "start_time_in_event": t0 + step * i} for i in range(n)]


def test_segment_ids_break_on_event_and_gap():
    meta = _meta(3, "a") + _meta(2, "b") + _meta(2, "b", t0=500.0)
    ids = segment_ids_from_meta(meta, max_gap=7.5)
    assert ids.tolist() == [0, 0, 0, 1, 1, 2, 2]


def test_segment_ids_missing_meta_returns_none():
    assert segment_ids_from_meta(None, 7.5) is None
    assert segment_ids_from_meta([{"event_id": "a"}], 7.5) is None


def test_label_vs_time_mode_differ_at_label_boundary_inside_segment():
    # One time-contiguous segment whose true label flips at index 3.
    probs = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    y = np.array([0, 0, 0, 1, 1, 1])
    ids = segment_ids_from_meta(_meta(6), max_gap=7.5)
    assert ids.tolist() == [0] * 6
    lab = moving_average_segmented(probs, y, 3)
    tim = moving_average_by_segments(probs, ids, 3)
    assert lab[3] == 1.0                      # label mode never crosses the boundary
    assert abs(tim[3] - 1 / 3) < 1e-12        # time mode smooths across it
    assert abs(tim[4] - 2 / 3) < 1e-12


def test_time_mode_ignores_labels_and_scattered_windows_are_noop():
    probs = np.array([0.1, 0.9, 0.2, 0.8])
    meta = [{"event_id": "a", "start_time_in_event": t} for t in (100.0, 5.0, 900.0, 50.0)]
    ids = segment_ids_from_meta(meta, 7.5)
    assert len(set(ids.tolist())) == 4
    assert np.allclose(moving_average_by_segments(probs, ids, 3), probs)


def _fake_pkl(td, with_meta):
    rng = np.random.default_rng(0)
    y = np.array([0] * 30 + [1] * 10)
    meta = _meta(30, "i") + _meta(10, "p")
    inner = []
    for _ in range(2):
        inner.append({
            "best_val_auc": 0.8,
            "val_probs": np.clip(rng.uniform(size=30), 1e-3, 1 - 1e-3),
            "val_labels": np.array([0] * 20 + [1] * 10),
            "test_probs": np.clip(rng.uniform(size=40), 1e-3, 1 - 1e-3),
            "test_labels": y.tolist(),
            **({"test_meta": meta} if with_meta else {}),
        })
    with open(os.path.join(td, "raw_predictions.pkl"), "wb") as f:
        pickle.dump({"outer_folds": [{"outer_fold": 1, "y_test": y.tolist(), "inner_folds": inner}]}, f)


def test_sweep_time_mode_writes_separate_dir_and_label_default_unchanged():
    with tempfile.TemporaryDirectory() as td:
        _fake_pkl(td, with_meta=True)
        kw = dict(calibration_methods=["none"], ma_windows=[3], thresholds=[0.5], make_plots=False)
        analyze_nested_calibration(td, **kw)
        s = analyze_nested_calibration(td, ma_segmentation="time", **kw)
        assert s["n_variants"] == 1
        assert os.path.exists(os.path.join(td, "analysis", "variant_summary.csv"))
        assert os.path.exists(os.path.join(td, "analysis_ma_time", "variant_summary.csv"))


def test_sweep_time_mode_without_meta_does_not_fall_back_to_labels():
    with tempfile.TemporaryDirectory() as td:
        _fake_pkl(td, with_meta=False)
        s = analyze_nested_calibration(td, ma_segmentation="time", calibration_methods=["none"],
                                       ma_windows=[3], thresholds=[0.5], make_plots=False)
        assert s["status"] == "missing_meta_for_time_segmentation"
        assert not os.path.exists(os.path.join(td, "analysis_ma_time"))
