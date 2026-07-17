from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from seizure_pred.analysis.metrics import moving_average_segmented, clinical_metrics
from seizure_pred.analysis.summary import analyze_multi_split_summary


def test_moving_average_segmented():
    probs = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.1])
    y = np.array([0, 0, 0, 1, 1, 0])
    
    # window_size = 3
    # region 1: 0 to 3 (indices 0, 1, 2) labels [0,0,0]
    # smoothed:
    # index 0: mean([0.1]) = 0.1
    # index 1: mean([0.1, 0.2]) = 0.15
    # index 2: mean([0.1, 0.2, 0.3]) = 0.2
    
    # region 2: 3 to 5 (indices 3, 4) labels [1,1]
    # region_len = 2 < 3, so:
    # index 3: win_start = 3, win_end = 4 => mean([0.8]) = 0.8
    # index 4: win_start = 3, win_end = 5 => mean([0.8, 0.9]) = 0.85
    
    # region 3: 5 to 6 (index 5) labels [0]
    # region_len = 1 < 3, so:
    # index 5: win_start = 5, win_end = 6 => mean([0.1]) = 0.1
    
    expected = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.1])
    res = moving_average_segmented(probs, y, window_size=3)
    np.testing.assert_allclose(res, expected)


def test_clinical_metrics():
    # Test case 1: Has preictal and positive prediction
    y_true = np.array([0, 0, 1, 1, 0])
    y_pred = np.array([0, 0, 0, 1, 0])
    
    metrics = clinical_metrics(y_true, y_pred, sampling_period=5.0)
    assert metrics["sensitivity"] == 1.0
    assert metrics["fpr_per_hour"] == 0.0

    # Test case 2: Has preictal and NO positive prediction
    y_pred = np.array([0, 0, 0, 0, 0])
    metrics = clinical_metrics(y_true, y_pred, sampling_period=5.0)
    assert metrics["sensitivity"] == 0.0
    assert metrics["fpr_per_hour"] == 0.0

    # Test case 3: False positives present
    y_pred = np.array([1, 0, 0, 0, 0])
    metrics = clinical_metrics(y_true, y_pred, sampling_period=5.0)
    assert metrics["sensitivity"] == 0.0
    # false positives = 1, interictal hours = 15 / 3600
    # fpr_per_hour = 1 / (15/3600) = 240.0
    assert metrics["fpr_per_hour"] == 240.0


def test_analyze_multi_split_summary():
    with tempfile.TemporaryDirectory() as td:
        run_dir = Path(td)
        
        # Create mock split directories
        for split in (0, 1):
            split_dir = run_dir / f"split_{split}"
            split_dir.mkdir()
            
            # Write mock predictions.jsonl
            preds_file = split_dir / "predictions.jsonl"
            rows = [
                {"y_true": 0, "prob": 0.1, "y_pred": 0},
                {"y_true": 0, "prob": 0.2, "y_pred": 0},
                {"y_true": 1, "prob": 0.9, "y_pred": 1},
                {"y_true": 1, "prob": 0.8, "y_pred": 1},
            ]
            with open(preds_file, "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
                    
        # Run summary analysis
        summary_data = analyze_multi_split_summary(
            str(run_dir),
            ma_windows=[1, 3],
            thresholds=[0.5],
            sampling_period=5.0,
            make_plots=True,
        )
        
        assert summary_data["n_splits"] == 2
        assert len(summary_data["sweep_variants"]) == 2
        
        # Check files were written
        assert (run_dir / "analysis" / "analysis_summary.json").exists()
        assert (run_dir / "analysis" / "pareto_optimal_variants.csv").exists()
        assert (run_dir / "analysis" / "plots" / "pareto_frontier.png").exists()
