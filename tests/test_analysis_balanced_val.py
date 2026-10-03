import numpy as np
import pytest

from seizure_pred.analysis.calibration_sweep import (
    balanced_metrics,
    best_f1_threshold,
    select_val_threshold,
)


def test_balanced_metrics_hand_computed():
    # 2 positives, 8 negatives -> negative weight = 2/8 = 0.25
    y = np.array([1, 1] + [0] * 8)
    pred = np.array([1, 0] + [1, 1] + [0] * 6)
    # tp=1 fn=1 ; fp=2*0.25=0.5 ; tn=6*0.25=1.5
    m = balanced_metrics(y, pred)
    assert m["balanced_precision"] == pytest.approx(1 / 1.5)
    assert m["balanced_recall"] == pytest.approx(0.5)
    assert m["balanced_specificity"] == pytest.approx(0.75)
    p, r = 1 / 1.5, 0.5
    assert m["balanced_f1"] == pytest.approx(2 * p * r / (p + r))


def test_balanced_metrics_single_class_nan():
    m = balanced_metrics(np.zeros(5), np.zeros(5))
    assert all(np.isnan(v) for v in m.values())


def test_best_f1_threshold():
    labels = np.array([0, 0, 0, 0, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.9])
    # F1 is maximal for any threshold in (0.35, 0.4]; the 0.01 grid gives 0.36..0.40
    assert 0.36 <= best_f1_threshold(probs, labels) <= 0.40
    assert np.isnan(best_f1_threshold(probs, np.zeros(6)))


def test_select_val_threshold_auc_weighted():
    labels = np.array([0, 0, 1, 1])
    f_a = {"val_probs": [0.1, 0.2, 0.3, 0.4], "val_labels": labels, "best_val_auc": 0.9}  # -> 0.3
    f_b = {"val_probs": [0.5, 0.6, 0.7, 0.8], "val_labels": labels, "best_val_auc": 0.1}  # -> 0.7
    ta, tb = best_f1_threshold(f_a["val_probs"], labels), best_f1_threshold(f_b["val_probs"], labels)
    assert select_val_threshold([f_a, f_b]) == pytest.approx(0.9 * ta + 0.1 * tb)
    assert np.isnan(select_val_threshold([{"best_val_auc": 0.5}]))
