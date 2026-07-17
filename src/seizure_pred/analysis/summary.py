from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .runs import load_predictions
from .metrics import moving_average_segmented, clinical_metrics
from ..core.runs import find_splits


def _mpl():
    # Lazy import of plotting libraries to avoid dependencies in CLI import time
    import matplotlib
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import seaborn as sns
    # Set premium styles
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["font.family"] = "sans-serif"
    return plt, sns


def analyze_multi_split_summary(
    run_dir: str,
    *,
    out_dir: Optional[str] = None,
    ma_windows: Optional[List[int]] = None,
    thresholds: Optional[List[float]] = None,
    sampling_period: float = 5.0,
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Run aggregate clinical sweeps and Pareto frontier analysis across all splits.

    Reads predictions.jsonl from each split_X folder under run_dir.
    Writes outputs to out_dir/analysis/ (default: run_dir/analysis/).
    """
    run_path = Path(run_dir)
    if out_dir is None:
        out_dir = str(run_path / "analysis")
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Find splits and load predictions
    splits = find_splits(str(run_path))
    if not splits:
        print(f"[analysis summary] No split folders found under {run_dir}")
        return {}

    # Read predictions for each split
    split_predictions = []
    for split_idx, split_dir in splits:
        preds_file = Path(split_dir) / "predictions.jsonl"
        if not preds_file.exists():
            continue
        try:
            y_true, prob, _, _ = load_predictions(str(preds_file))
            if y_true.size > 0:
                split_predictions.append({
                    "split_idx": split_idx,
                    "y_true": y_true,
                    "prob": prob
                })
        except Exception as e:
            print(f"[analysis summary] Failed to load predictions for split_{split_idx}: {e}")

    if not split_predictions:
        print(f"[analysis summary] No valid predictions found in splits under {run_dir}")
        return {}

    # 2. Setup sweep parameters
    if ma_windows is None:
        ma_windows = [1, 3, 5, 7, 10]
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print(f"[analysis summary] Analyzing {len(split_predictions)} splits:")
    print(f"  MA windows: {ma_windows}")
    print(f"  Thresholds: {thresholds}")
    print(f"  Sampling period: {sampling_period}s")

    # 3. Perform sweeps across splits
    from sklearn.metrics import roc_auc_score

    sweep_results = []
    for window in ma_windows:
        for threshold in thresholds:
            variant_name = f"ma_{window}_thr_{threshold:.2f}"
            
            fold_metrics = []
            for sp in split_predictions:
                y_true = sp["y_true"]
                prob = sp["prob"]
                
                # Apply moving average
                smoothed_prob = moving_average_segmented(prob, y_true, window)
                
                # Apply threshold
                y_pred = (smoothed_prob >= threshold).astype(int)
                
                # Calculate metrics
                clin = clinical_metrics(y_true, y_pred, sampling_period=sampling_period)
                
                # Calculate AUC on smoothed probabilities
                try:
                    auc = float(roc_auc_score(y_true, smoothed_prob))
                except Exception:
                    auc = float("nan")
                
                fold_metrics.append({
                    "auc": auc,
                    "sensitivity": clin["sensitivity"],
                    "fpr_per_hour": clin["fpr_per_hour"]
                })
            
            # Aggregate across folds
            aucs = [m["auc"] for m in fold_metrics if not np.isnan(m["auc"])]
            sensitivities = [m["sensitivity"] for m in fold_metrics if not np.isnan(m["sensitivity"])]
            fprs = [m["fpr_per_hour"] for m in fold_metrics if not np.isnan(m["fpr_per_hour"])]
            
            mean_auc = float(np.mean(aucs)) if aucs else float("nan")
            std_auc = float(np.std(aucs)) if aucs else float("nan")
            
            mean_sens = float(np.mean(sensitivities)) if sensitivities else float("nan")
            std_sens = float(np.std(sensitivities)) if sensitivities else float("nan")
            
            mean_fpr = float(np.mean(fprs)) if fprs else float("nan")
            std_fpr = float(np.std(fprs)) if fprs else float("nan")
            
            sweep_results.append({
                "variant": variant_name,
                "config_ma_window": window,
                "config_threshold": threshold,
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "mean_sensitivity": mean_sens,
                "std_sensitivity": std_sens,
                "mean_fpr_per_hour": mean_fpr,
                "std_fpr_per_hour": std_fpr,
            })

    summary_df = pd.DataFrame(sweep_results)
    
    # 4. Identify Pareto-optimal frontier (Maximize sensitivity, Minimize FPR per hour)
    valid_data = summary_df[
        ~summary_df["mean_sensitivity"].isna() & 
        ~summary_df["mean_fpr_per_hour"].isna()
    ].copy()

    pareto_indices = []
    for idx, row in valid_data.iterrows():
        is_pareto = True
        for _, other_row in valid_data.iterrows():
            if (other_row["mean_sensitivity"] >= row["mean_sensitivity"] and
                other_row["mean_fpr_per_hour"] <= row["mean_fpr_per_hour"] and
                (other_row["mean_sensitivity"] > row["mean_sensitivity"] or
                 other_row["mean_fpr_per_hour"] < row["mean_fpr_per_hour"])):
                is_pareto = False
                break
        if is_pareto:
            pareto_indices.append(idx)
            
    pareto_df = valid_data.loc[pareto_indices].sort_values("mean_fpr_per_hour")
    
    # Save Pareto CSV
    pareto_csv_path = out_path / "pareto_optimal_variants.csv"
    pareto_df.to_csv(pareto_csv_path, index=False)
    
    # Save Full JSON summary
    summary_json_path = out_path / "analysis_summary.json"
    summary_data = {
        "run_dir": str(run_path),
        "sampling_period": sampling_period,
        "n_splits": len(split_predictions),
        "sweep_variants": sweep_results,
        "pareto_variants": pareto_df.to_dict(orient="records"),
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"[analysis summary] Summary written to {summary_json_path}")
    print(f"[analysis summary] Pareto optimal variants written to {pareto_csv_path} ({len(pareto_df)} configurations)")

    # 5. Generate plots
    if make_plots and len(valid_data) > 0:
        try:
            plots_dir = out_path / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            plt, sns = _mpl()
            
            # --- Plot 1: Pareto Frontier ---
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(
                valid_data["mean_fpr_per_hour"],
                valid_data["mean_sensitivity"],
                c=valid_data["mean_auc"],
                cmap="viridis",
                s=70,
                alpha=0.5,
                edgecolors="none"
            )
            ax.plot(
                pareto_df["mean_fpr_per_hour"],
                pareto_df["mean_sensitivity"],
                color="#e056fd",
                linestyle="-",
                linewidth=2.5,
                marker="o",
                markersize=8,
                label="Pareto Frontier"
            )
            
            cbar = fig.colorbar(scatter, ax=ax)
            cbar.set_label("Mean AUC", fontsize=11)
            ax.set_xlabel("Mean False Positive Rate per Hour", fontsize=12)
            ax.set_ylabel("Mean Sensitivity (Event-Level)", fontsize=12)
            ax.set_title("Sensitivity vs FPR Tradeoff (Pareto Frontier)", fontsize=14, fontweight="bold", pad=15)
            ax.legend(frameon=True, facecolor="white", edgecolor="none")
            fig.tight_layout()
            fig.savefig(plots_dir / "pareto_frontier.png", bbox_inches="tight", dpi=300)
            plt.close(fig)

            # --- Plot 2: Threshold Sensitivity Analysis (per MA window) ---
            for window in ma_windows:
                df_win = valid_data[valid_data["config_ma_window"] == window].sort_values("config_threshold")
                if len(df_win) == 0:
                    continue
                
                fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
                
                # AUC
                axes[0].plot(df_win["config_threshold"], df_win["mean_auc"], "o-", color="#1e90ff", linewidth=2.5, markersize=6)
                axes[0].set_xlabel("Threshold", fontsize=11)
                axes[0].set_ylabel("Mean AUC", fontsize=11)
                axes[0].set_title("Mean AUC vs Threshold", fontsize=12, fontweight="bold")
                axes[0].set_ylim(0.0, 1.05)
                
                # Sensitivity
                axes[1].plot(df_win["config_threshold"], df_win["mean_sensitivity"], "o-", color="#2ed573", linewidth=2.5, markersize=6)
                axes[1].set_xlabel("Threshold", fontsize=11)
                axes[1].set_ylabel("Mean Sensitivity", fontsize=11)
                axes[1].set_title("Mean Sensitivity vs Threshold", fontsize=12, fontweight="bold")
                axes[1].set_ylim(-0.05, 1.05)
                
                # FPR/hour
                axes[2].plot(df_win["config_threshold"], df_win["mean_fpr_per_hour"], "o-", color="#ff4757", linewidth=2.5, markersize=6)
                axes[2].set_xlabel("Threshold", fontsize=11)
                axes[2].set_ylabel("Mean FPR/hour", fontsize=11)
                axes[2].set_title("Mean FPR/hour vs Threshold", fontsize=12, fontweight="bold")
                
                fig.suptitle(f"Threshold Sensitivity Analysis (MA Window = {window})", fontsize=15, fontweight="bold")
                fig.tight_layout()
                fig.savefig(plots_dir / f"threshold_analysis_ma{window}.png", bbox_inches="tight", dpi=300)
                plt.close(fig)

            # --- Plot 3: Metrics vs MA Window (for selected thresholds) ---
            selected_thresholds = [0.3, 0.5, 0.7]
            fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
            
            colors = ["#1e90ff", "#2ed573", "#ff4757"]
            
            for i, thr in enumerate(selected_thresholds):
                df_thr = valid_data[valid_data["config_threshold"] == thr].sort_values("config_ma_window")
                if len(df_thr) == 0:
                    continue
                axes[0].plot(df_thr["config_ma_window"], df_thr["mean_auc"], "o-", label=f"thr={thr:.2f}", color=colors[i % len(colors)], linewidth=2.0)
                axes[1].plot(df_thr["config_ma_window"], df_thr["mean_sensitivity"], "o-", label=f"thr={thr:.2f}", color=colors[i % len(colors)], linewidth=2.0)
                axes[2].plot(df_thr["config_ma_window"], df_thr["mean_fpr_per_hour"], "o-", label=f"thr={thr:.2f}", color=colors[i % len(colors)], linewidth=2.0)
                
            axes[0].set_title("Mean AUC vs Window Size", fontsize=12, fontweight="bold")
            axes[0].set_xlabel("MA Window Size", fontsize=11)
            axes[0].set_ylabel("AUC", fontsize=11)
            axes[0].legend()
            
            axes[1].set_title("Mean Sensitivity vs Window Size", fontsize=12, fontweight="bold")
            axes[1].set_xlabel("MA Window Size", fontsize=11)
            axes[1].set_ylabel("Sensitivity", fontsize=11)
            axes[1].set_ylim(-0.05, 1.05)
            axes[1].legend()
            
            axes[2].set_title("Mean FPR/hour vs Window Size", fontsize=12, fontweight="bold")
            axes[2].set_xlabel("MA Window Size", fontsize=11)
            axes[2].set_ylabel("FPR/hour", fontsize=11)
            axes[2].legend()
            
            fig.suptitle("Smoothing Window size Comparison", fontsize=15, fontweight="bold")
            fig.tight_layout()
            fig.savefig(plots_dir / "metrics_vs_window.png", bbox_inches="tight", dpi=300)
            plt.close(fig)
            
            print(f"[analysis summary] Generated summary plots in {plots_dir}")

        except Exception as e:
            print(f"[analysis summary] Failed to generate sweep plots: {e}")

    return summary_data
