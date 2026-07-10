"""Tests for newly added/fixed features: AUC, validation, grid, calibration,
CV methods, suppression FPR, benchmark, and the nested calibration sweep."""
from __future__ import annotations

import json
import os
import pickle
import tempfile

import numpy as np
import pytest
import torch


def _set_threads():
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


# --------------------------------------------------------------------------- AUC
def test_binary_metrics_compute_auc_and_ap():
    from seizure_pred.training.engine.metrics import binary_classification_metrics

    logits = torch.tensor([0.1, 0.2, 2.9, 3.1, -0.5, 3.5])
    targets = torch.tensor([0, 0, 1, 1, 0, 1])
    m = binary_classification_metrics(logits, targets)
    assert "auc" in m and "ap" in m
    # Perfect-ish separation -> AUC close to 1.0
    assert m["auc"] > 0.9
    assert 0.0 <= m["ap"] <= 1.0


def test_auc_known_value():
    from seizure_pred.training.engine.metrics import binary_auc

    probs = np.array([0.1, 0.4, 0.35, 0.8])
    y = np.array([0, 0, 1, 1])
    assert abs(binary_auc(probs, y) - 0.75) < 1e-9


# ---------------------------------------------------------------- validation
def test_validate_rejects_unknown_nested_key():
    from seizure_pred.core.validate import validate_config_dict, ConfigValidationError
    from seizure_pred.core.config import TrainConfig

    bad = {"data": {"name": "synthetic", "n_fold": 5}}  # wrong field name
    with pytest.raises(ConfigValidationError):
        validate_config_dict(bad, TrainConfig)


def test_validate_accepts_correct_nested_and_postprocess():
    from seizure_pred.core.validate import validate_config_dict
    from seizure_pred.core.config import TrainConfig
    from seizure_pred.core.io import from_dict

    good = {
        "device": "cpu", "epochs": 1, "task": "prediction",
        "monitor": "auc", "monitor_mode": "max",
        "data": {"name": "synthetic", "n_folds": 2, "kwargs": {"n": 16, "c": 8, "t": 32, "pos_frac": 0.5, "seed": 1}},
        "postprocess": {"name": "threshold", "kwargs": {"threshold": 0.6}},
    }
    validate_config_dict(good, TrainConfig)  # should not raise
    cfg = from_dict(TrainConfig, good)
    assert cfg.monitor == "auc"
    assert cfg.postprocess is not None and cfg.postprocess.name == "threshold"
    assert cfg.postprocess.kwargs["threshold"] == 0.6


# ---------------------------------------------------------------- CV nearest
def test_leave_one_out_nearest_method():
    from seizure_pred.training.datasets.synthetic import SyntheticEEGDataset
    from seizure_pred.data.splits import leave_one_out

    ds = SyntheticEEGDataset(n=64, c=4, t=16, pos_frac=0.25, seed=1)
    folds = list(leave_one_out(ds, method="nearest", random_state=0))
    assert len(folds) >= 1
    for train_set, test_set in folds:
        # positives appear in test, no overlap between train/test base indices
        assert len(test_set) > 0
        train_base = set(train_set.base_indices.tolist())
        test_base = set(test_set.base_indices.tolist())
        assert train_base.isdisjoint(test_base)
        # test must contain the held-out positive group
        assert int(test_set.y.sum()) > 0


def test_leave_one_out_balanced_vs_nearest_partition():
    from seizure_pred.training.datasets.synthetic import SyntheticEEGDataset
    from seizure_pred.data.splits import leave_one_out

    ds = SyntheticEEGDataset(n=64, c=4, t=16, pos_frac=0.25, seed=2)
    bal = list(leave_one_out(ds, method="balanced", random_state=0))
    near = list(leave_one_out(ds, method="nearest", random_state=0))
    assert len(bal) == len(near)


# ---------------------------------------------------------------- suppression FPR
def test_clinical_metrics_suppression():
    from seizure_pred.analysis.metrics import clinical_metrics

    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
    # 3 consecutive false alarms then a true detection
    y_pred = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    m = clinical_metrics(y_true, y_pred, sampling_period=5.0, suppression_duration=2)
    assert m["sensitivity"] == 1.0
    # without suppression: 3 FPs over 8 interictal windows * 5s = 40s -> 3 FP / (40/3600) hours
    assert m["fpr_per_hour"] > 0
    # with suppression the first run of 3 alarms collapses to 1 effective alarm
    assert m["fpr_per_hour_suppressed"] < m["fpr_per_hour"]


# ---------------------------------------------------------------- calibration
def test_probability_calibrator_methods():
    from seizure_pred.inference.calibration import ProbabilityCalibrator

    rng = np.random.default_rng(0)
    n = 200
    val_probs = np.clip(np.concatenate([
        rng.uniform(0.0, 0.4, n // 2), rng.uniform(0.55, 1.0, n // 2)]), 1e-4, 1 - 1e-4)
    val_labels = np.array([0] * (n // 2) + [1] * (n // 2))

    for method in ["percentile", "beta", "isotonic", "temperature"]:
        cal = ProbabilityCalibrator(method=method)
        cal.fit(val_probs, val_labels)
        out = cal.transform(val_probs)
        assert out.shape == val_probs.shape
        assert np.all(out >= 0) and np.all(out <= 1)


def test_calibrate_ensemble_shapes():
    from seizure_pred.inference.calibration import calibrate_ensemble

    rng = np.random.default_rng(1)
    test_stack = rng.uniform(0, 1, (3, 50))
    val_probs_list = [rng.uniform(0, 1, 40) for _ in range(3)]
    val_labels_list = [rng.integers(0, 2, 40) for _ in range(3)]
    val_aucs = np.array([0.7, 0.8, 0.6])
    final, cals = calibrate_ensemble(test_stack, val_probs_list, val_labels_list, val_aucs,
                                     calibration_method="percentile")
    assert final.shape == (50,)
    assert len(cals) == 3


# ---------------------------------------------------------------- grid runner
def test_run_grid_synthetic():
    _set_threads()
    import yaml
    from seizure_pred.experiments.grid import run_grid

    with tempfile.TemporaryDirectory() as td:
        cfg_data = {
            "device": "cpu", "amp": False, "epochs": 1, "task": "prediction",
            "save_dir": td, "run_name": "grid_test",
            "data": {"name": "synthetic", "batch_size": 4, "num_workers": 0,
                     "pin_memory": False, "persistent_workers": False,
                     "split_method": "stratified", "n_folds": 2,
                     "dataloader_type": "torch",
                     "kwargs": {"n": 32, "c": 8, "t": 32, "pos_frac": 0.5, "seed": 1}},
            "model": {"name": "simple_cnn", "num_classes": 1, "in_channels": 8, "kwargs": {"hidden": 4}},
        }
        cfg_file = os.path.join(td, "cfg.yaml")
        with open(cfg_file, "w") as f:
            yaml.dump(cfg_data, f)
        run_dirs = run_grid(cfg_file, {"optim.lr": [1e-3, 3e-4]}, split_index=0, n_folds=2)
        assert len(run_dirs) == 2
        for rd in run_dirs:
            assert os.path.exists(os.path.join(rd, "checkpoints", "best.pt"))


# ---------------------------------------------------------------- benchmark
def test_benchmark_simple_cnn():
    _set_threads()
    from seizure_pred.experiments.benchmark import ModelBenchmark
    import seizure_pred.models as models

    models.register_all()
    bench = ModelBenchmark("simple_cnn", input_shape=(1, 8, 64), model_kwargs={"hidden": 4})
    res = bench.benchmark(n_runs=2, batch_size=1)
    assert res["total_params"] > 0
    assert res["cpu_mean_ms"] > 0


# ---------------------------------------------------------------- nested calibration sweep
def test_analyze_nested_calibration_on_fake_pkl():
    from seizure_pred.analysis.calibration_sweep import analyze_nested_calibration

    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(3)
        outer_folds = []
        for _ in range(2):
            n_test = 40
            y_test = np.array([0] * 30 + [1] * 10)
            inner_folds = []
            for _ in range(2):
                val_labels = np.array([0] * 20 + [1] * 10)
                val_probs = np.clip(np.concatenate([
                    rng.uniform(0, 0.4, 20), rng.uniform(0.5, 1.0, 10)]), 1e-4, 1 - 1e-4)
                test_probs = np.clip(np.concatenate([
                    rng.uniform(0, 0.45, 30), rng.uniform(0.55, 1.0, 10)]), 1e-4, 1 - 1e-4)
                inner_folds.append({
                    "best_val_auc": 0.8,
                    "val_probs": val_probs, "val_labels": val_labels,
                    "test_probs": test_probs, "test_labels": y_test.tolist(),
                })
            outer_folds.append({"outer_fold": 1, "y_test": y_test.tolist(), "inner_folds": inner_folds})

        with open(os.path.join(td, "raw_predictions.pkl"), "wb") as f:
            pickle.dump({"outer_folds": outer_folds}, f)

        summary = analyze_nested_calibration(
            td,
            calibration_methods=["none", "percentile"],
            ma_windows=[1, 3],
            thresholds=[0.5],
            percentiles=[10],
            sampling_period=5.0,
            make_plots=False,
        )
        assert summary["n_outer_folds"] == 2
        # variants = 2 cal (none + percentile_p10) x 2 ma x 1 thr = 4
        assert summary["n_variants"] == 4
        assert os.path.exists(os.path.join(td, "analysis", "variant_summary.csv"))
        assert os.path.exists(os.path.join(td, "analysis", "pareto_optimal_variants.csv"))
        # F1 is now computed -> best_variants_f1.csv is produced
        assert os.path.exists(os.path.join(td, "analysis", "best_variants_f1.csv"))
        assert os.path.exists(os.path.join(td, "analysis", "best_variants_auc.csv"))


# ---------------------------------------------------------------- monitor validation
def test_monitor_validation_rejects_unknown_metric():
    _set_threads()
    from seizure_pred.core.config import TrainConfig
    from seizure_pred.training.engine.pipeline import build_dataset, iter_splits
    from seizure_pred.training.registries import MODELS, LOSSES, OPTIMIZERS
    import seizure_pred.models as models
    from seizure_pred.training.engine.trainer import Trainer

    models.register_all()
    cfg = TrainConfig()
    cfg.device = "cpu"
    cfg.amp = False
    cfg.epochs = 1
    cfg.monitor = "not_a_real_metric"
    cfg.data.name = "synthetic"
    cfg.data.kwargs = {"n": 16, "c": 8, "t": 32, "pos_frac": 0.5, "seed": 1}
    cfg.model.name = "simple_cnn"
    cfg.model.in_channels = 8

    ds = build_dataset(cfg)
    train_set, val_set = next(iter(iter_splits(ds, cfg.data)))
    model = MODELS.create(cfg.model.name, cfg.model)
    with pytest.raises(ValueError, match="Unknown monitor metric"):
        Trainer(
            model=model,
            loss_fn=LOSSES.create("bce_logits"),
            optimizer=OPTIMIZERS.create("adam", model.parameters(), lr=1e-3),
            scheduler=None, cfg=cfg, run_dir="runs/_monitor_test",
        )
