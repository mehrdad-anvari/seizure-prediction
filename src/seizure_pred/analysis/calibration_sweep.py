"""Calibration-aware analysis sweep for nested-CV runs.

This module mirrors the legacy ``analyze_results2.py`` / ``analyze_results3.py``
post-processing: given a ``raw_predictions.pkl`` produced by nested
cross-validation, it explores a grid of

    calibration method x moving-average window x threshold

and reports per-variant clinical metrics (AUC, sensitivity, FPR/hour,
optionally suppression-based FPR), a Pareto frontier, and comparison CSVs.

The raw-pickle schema (produced by ``seizure_pred.cli.train_cmd.run_nested_cv``)::

    {
      "outer_folds": [
        {
          "outer_fold": int,
          "y_test": [...],
          "inner_folds": [
            {
              "best_val_auc": float,
              "val_probs": [...], "val_labels": [...],
              "test_probs": [...], "test_labels": [...],
              ...
            }, ...
          ]
        }, ...
      ]
    }
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .metrics import moving_average_segmented, clinical_metrics
from ..inference.calibration import ProbabilityCalibrator

try:
    from sklearn.metrics import roc_auc_score as _sk_auc
except Exception:  # pragma: no cover
    _sk_auc = None  # type: ignore


def _auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if _sk_auc is not None:
        try:
            return float(_sk_auc(y_true, prob))
        except Exception:
            pass
    from ..training.engine.metrics import binary_auc

    return float(binary_auc(prob, y_true))


def _ensemble_auc_weighted(test_probs_stack: np.ndarray, val_aucs: np.ndarray) -> np.ndarray:
    weights = np.asarray(val_aucs, dtype=np.float64)
    if weights.size == 0 or not np.isfinite(weights).any() or weights.sum() <= 0:
        weights = np.ones(test_probs_stack.shape[0]) / test_probs_stack.shape[0]
    else:
        weights = np.clip(weights, 0, None)
        weights = weights / weights.sum()
    return np.tensordot(weights, test_probs_stack, axes=1)


def _calibrate_fold(
    test_probs_stack: np.ndarray,
    val_probs_list: List[np.ndarray],
    val_labels_list: List[np.ndarray],
    val_aucs: np.ndarray,
    method: str,
    percentile: Optional[int] = None,
) -> np.ndarray:
    """Return (n_test,) calibrated + AUC-weight-ensembled test probabilities."""
    if method == "none":
        return _ensemble_auc_weighted(test_probs_stack, val_aucs)

    kwargs: Dict[str, Any] = {}
    if method == "percentile" and percentile is not None:
        kwargs["target_preictal_percentile"] = int(percentile)

    n_models = test_probs_stack.shape[0]
    calibrated = []
    for i in range(n_models):
        cal = ProbabilityCalibrator(method=method, **kwargs)
        cal.fit(val_probs_list[i], val_labels_list[i])
        calibrated.append(cal.transform(test_probs_stack[i]))
    calibrated_stack = np.stack(calibrated)
    weights = np.asarray(val_aucs, dtype=np.float64)
    if weights.size == 0 or not np.isfinite(weights).any() or weights.sum() <= 0:
        weights = np.ones(n_models) / n_models
    else:
        weights = np.clip(weights, 0, None)
        weights = weights / weights.sum()
    return np.tensordot(weights, calibrated_stack, axes=1)


def analyze_nested_calibration(
    run_dir: str,
    *,
    out_dir: Optional[str] = None,
    calibration_methods: Optional[List[str]] = None,
    ma_windows: Optional[List[int]] = None,
    thresholds: Optional[List[float]] = None,
    percentiles: Optional[List[int]] = None,
    sampling_period: float = 5.0,
    suppression_duration: Optional[int] = None,
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Run the calibration x MA x threshold sweep over a nested-CV run.

    Reads ``<run_dir>/raw_predictions.pkl`` and writes reports/CSVs/plots under
    ``<out_dir>`` (default ``<run_dir>/analysis``).
    """
    run_path = Path(run_dir)
    pkl_path = run_path / "raw_predictions.pkl"
    if not pkl_path.exists():
        return {"status": "no_raw_predictions", "run_dir": str(run_path)}

    with open(pkl_path, "rb") as f:
        cv_results = pickle.load(f)

    if out_dir is None:
        out_dir = str(run_path / "analysis")
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if calibration_methods is None:
        calibration_methods = ["none", "percentile", "beta", "isotonic", "temperature"]
    if ma_windows is None:
        ma_windows = [1, 3, 5, 7, 10]
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    if percentiles is None:
        percentiles = [5, 10, 15, 20]

    outer_folds = cv_results.get("outer_folds", [])
    if not outer_folds:
        return {"status": "empty_outer_folds", "run_dir": str(run_path)}

    # Pre-extract per-fold arrays
    fold_data = []
    for of in outer_folds:
        inner = of.get("inner_folds", [])
        if not inner:
            continue
        test_probs_stack = np.array([np.asarray(inf["test_probs"], dtype=np.float64) for inf in inner])
        val_probs_list = [np.asarray(inf["val_probs"], dtype=np.float64) for inf in inner]
        val_labels_list = [np.asarray(inf["val_labels"], dtype=np.int64) for inf in inner]
        val_aucs = np.array([float(inf.get("best_val_auc", 0.5)) for inf in inner], dtype=np.float64)
        y_test = np.asarray(of.get("y_test"), dtype=np.int64)
        if y_test.size == 0 and inner:
            y_test = np.asarray(inner[0].get("test_labels"), dtype=np.int64)
        # Align lengths (defensive)
        min_len = min(test_probs_stack.shape[1], y_test.size)
        test_probs_stack = test_probs_stack[:, :min_len]
        y_test = y_test[:min_len]
        fold_data.append({
            "test_probs_stack": test_probs_stack,
            "val_probs_list": val_probs_list,
            "val_labels_list": val_labels_list,
            "val_aucs": val_aucs,
            "y_test": y_test,
        })

    if not fold_data:
        return {"status": "no_usable_folds", "run_dir": str(run_path)}

    # Build variant list: (calibration_method, percentile_or_None)
    cal_variants: List[tuple] = []
    for m in calibration_methods:
        if m == "percentile":
            for p in percentiles:
                cal_variants.append((m, p))
        else:
            cal_variants.append((m, None))

    sweep_results: List[Dict[str, Any]] = []
    for method, percentile in cal_variants:
        cal_label = method if percentile is None else f"percentile_p{percentile}"
        # Calibrate per fold (returns n_test probs per fold)
        fold_calibrated = []
        for fd in fold_data:
            try:
                cal_probs = _calibrate_fold(
                    fd["test_probs_stack"], fd["val_probs_list"], fd["val_labels_list"],
                    fd["val_aucs"], method, percentile,
                )
            except Exception as e:
                print(f"[calibration] {cal_label} failed on a fold: {e}; falling back to none")
                cal_probs = _ensemble_auc_weighted(fd["test_probs_stack"], fd["val_aucs"])
            fold_calibrated.append((fd["y_test"], cal_probs))

        for window in ma_windows:
            for threshold in thresholds:
                fold_metrics = []
                for y_test, cal_probs in fold_calibrated:
                    smoothed = moving_average_segmented(cal_probs, y_test, window)
                    y_pred = (smoothed >= threshold).astype(int)
                    clin = clinical_metrics(y_test, y_pred, sampling_period=sampling_period,
                                            suppression_duration=suppression_duration)
                    try:
                        auc = _auc(y_test, smoothed)
                    except Exception:
                        auc = float("nan")
                    # window-level precision/recall/f1
                    tp = int(((y_test == 1) & (y_pred == 1)).sum())
                    fp = int(((y_test == 0) & (y_pred == 1)).sum())
                    fn = int(((y_test == 1) & (y_pred == 0)).sum())
                    prec = tp / max(1, tp + fp)
                    rec = tp / max(1, tp + fn)
                    f1 = 2 * prec * rec / max(1e-12, prec + rec)
                    fold_metrics.append({
                        "auc": auc,
                        "f1": float(f1),
                        "sensitivity": clin["sensitivity"],
                        "fpr_per_hour": clin["fpr_per_hour"],
                        "fpr_per_hour_suppressed": clin["fpr_per_hour_suppressed"],
                    })

                def _agg(key):
                    vals = [m[key] for m in fold_metrics if not np.isnan(m[key])]
                    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), float("nan"))

                mean_auc, std_auc = _agg("auc")
                mean_f1, std_f1 = _agg("f1")
                mean_sens, std_sens = _agg("sensitivity")
                mean_fpr, std_fpr = _agg("fpr_per_hour")
                mean_fpr_sup, std_fpr_sup = _agg("fpr_per_hour_suppressed")

                sweep_results.append({
                    "calibration": cal_label,
                    "config_calibration": cal_label,
                    "config_ma_window": window,
                    "config_threshold": threshold,
                    "mean_auc": mean_auc, "std_auc": std_auc,
                    "mean_f1": mean_f1, "std_f1": std_f1,
                    "mean_sensitivity": mean_sens, "std_sensitivity": std_sens,
                    "mean_fpr_per_hour": mean_fpr, "std_fpr_per_hour": std_fpr,
                    "mean_fpr_per_hour_suppressed": mean_fpr_sup, "std_fpr_per_hour_suppressed": std_fpr_sup,
                })

    summary_df = pd.DataFrame(sweep_results)

    # ---- Pareto frontier (maximize sensitivity, minimize fpr_per_hour) ----
    valid = summary_df[~summary_df["mean_sensitivity"].isna() & ~summary_df["mean_fpr_per_hour"].isna()].copy()
    pareto_idx = []
    for idx, row in valid.iterrows():
        dominated = False
        for _, other in valid.iterrows():
            if (other["mean_sensitivity"] >= row["mean_sensitivity"] and
                    other["mean_fpr_per_hour"] <= row["mean_fpr_per_hour"] and
                    (other["mean_sensitivity"] > row["mean_sensitivity"] or
                     other["mean_fpr_per_hour"] < row["mean_fpr_per_hour"])):
                dominated = True
                break
        if not dominated:
            pareto_idx.append(idx)
    pareto_df = valid.loc[pareto_idx].sort_values("mean_fpr_per_hour") if pareto_idx else valid.iloc[0:0]

    # ---- Write CSVs ----
    summary_df.to_csv(out_path / "variant_summary.csv", index=False)
    pareto_df.to_csv(out_path / "pareto_optimal_variants.csv", index=False)
    for metric in ("auc", "sensitivity", "f1", "fpr_per_hour"):
        col = f"mean_{metric}" if metric != "fpr_per_hour" else "mean_fpr_per_hour"
        if col in summary_df.columns:
            top = summary_df.sort_values(col, ascending=(metric == "fpr_per_hour")).head(10)
            top.to_csv(out_path / f"best_variants_{metric}.csv", index=False)
    if "config_calibration" in summary_df.columns:
        summary_df.groupby("config_calibration")[["mean_auc", "mean_sensitivity", "mean_fpr_per_hour"]].mean().to_csv(
            out_path / "calibration_comparison.csv"
        )
    summary_df.groupby("config_ma_window")[["mean_auc", "mean_sensitivity", "mean_fpr_per_hour"]].mean().to_csv(
        out_path / "ma_window_comparison.csv"
    )
    summary_df.groupby("config_threshold")[["mean_auc", "mean_sensitivity", "mean_fpr_per_hour"]].mean().to_csv(
        out_path / "threshold_comparison.csv"
    )

    summary_data = {
        "run_dir": str(run_path),
        "sampling_period": sampling_period,
        "suppression_duration": suppression_duration,
        "n_outer_folds": len(fold_data),
        "calibration_methods": calibration_methods,
        "ma_windows": ma_windows,
        "thresholds": thresholds,
        "percentiles": percentiles,
        "n_variants": len(sweep_results),
        "pareto_variants": pareto_df.to_dict(orient="records"),
    }
    with open(out_path / "calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[calibration] {len(sweep_results)} variants across {len(fold_data)} folds -> {out_path}")
    print(f"[calibration] Pareto-optimal variants: {len(pareto_df)}")

    # ---- Plots ----
    if make_plots and len(valid) > 0:
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt

            plots = out_path / "plots"
            plots.mkdir(parents=True, exist_ok=True)

            # Pareto frontier
            fig, ax = plt.subplots(figsize=(8, 6))
            sc = ax.scatter(valid["mean_fpr_per_hour"], valid["mean_sensitivity"],
                            c=valid["mean_auc"], cmap="viridis", s=45, alpha=0.5)
            if len(pareto_df):
                ax.plot(pareto_df["mean_fpr_per_hour"], pareto_df["mean_sensitivity"],
                        "o-", color="#e056fd", lw=2, ms=7, label="Pareto frontier")
            fig.colorbar(sc, ax=ax, label="Mean AUC")
            ax.set_xlabel("Mean FPR/hour")
            ax.set_ylabel("Mean Sensitivity")
            ax.set_title("Calibration sweep: Sensitivity vs FPR (Pareto)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(plots / "calibration_pareto_frontier.png", dpi=200, bbox_inches="tight")
            plt.close(fig)

            # Calibration method comparison boxplot
            fig, ax = plt.subplots(figsize=(10, 5))
            labels = sorted(summary_df["config_calibration"].unique())
            ax.boxplot([summary_df.loc[summary_df["config_calibration"] == l, "mean_auc"].dropna() for l in labels],
                       labels=labels)
            ax.set_ylabel("Mean AUC")
            ax.set_title("AUC by calibration method")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(plots / "calibration_comparison.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            print(f"[calibration] plot generation failed: {e}")

    return summary_data
