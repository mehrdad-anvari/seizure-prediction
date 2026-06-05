from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .runs import load_predictions
from .metrics import (
    binary_report,
    roc_curve,
    pr_curve,
    auc_trapz,
)
from .plots import (
    plot_history,
    plot_confusion,
    plot_roc,
    plot_pr,
)


@dataclass
class AnalyzeResult:
    report: Dict[str, Any]
    paths: Dict[str, str]


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_load_json(path: str) -> Optional[dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def analyze_run(
    run_dir: str,
    *,
    out_dir: Optional[str] = None,
    threshold: float = 0.5,
    prefer_postprocessed: bool = True,
    make_plots: bool = True,
) -> Dict[str, Any]:
    """Analyze a run directory and write plots/reports.

    Reads (if available):
      - config.json
      - history.jsonl
      - predictions.jsonl

    Writes under <out_dir>/analysis (default: <run_dir>/analysis):
      - history.png
      - confusion.png
      - roc.png
      - pr.png
      - report.json
      - report.txt
    """
    if out_dir is None:
        out_dir = os.path.join(run_dir, "analysis")
    _ensure_dir(out_dir)

    report: Dict[str, Any] = {
        "run_dir": run_dir,
        "out_dir": out_dir,
        "threshold": float(threshold),
        "prefer_postprocessed": bool(prefer_postprocessed),
        "status": {},
    }

    # ---- History plot (optional)
    history_path = os.path.join(run_dir, "history.jsonl")
    if make_plots and os.path.exists(history_path):
        try:
            hist_png = os.path.join(out_dir, "history.png")
            plot_history(history_path, save_path=hist_png)
            report["status"]["history"] = "ok"
            report["artifacts"] = report.get("artifacts", {})
            report["artifacts"]["history_plot"] = hist_png
        except Exception as e:
            report["status"]["history"] = f"failed: {e}"
    else:
        report["status"]["history"] = "missing" if not os.path.exists(history_path) else "skipped"

    # ---- Predictions (required for metrics/curves)
    preds_path = os.path.join(run_dir, "predictions.jsonl")
    if not os.path.exists(preds_path):
        # Some setups store predictions under analysis/eval folders; fall back to run_dir root only.
        report["status"]["predictions"] = "missing"
        _write_report_files(out_dir, report)
        return report

    y_true, prob, y_pred, y_pred_post = load_predictions(preds_path)

    if y_true.size == 0:
        report["status"]["predictions"] = "empty"
        _write_report_files(out_dir, report)
        return report

    # Empty predictions: degrade gracefully (still write report files)
    if y_true.size == 0:
        report["status"]["predictions"] = "empty"
        _write_report_files(out_dir, report)
        return report

    # Choose which labels to use for confusion/report
    if prefer_postprocessed and y_pred_post is not None:
        y_hat = y_pred_post
        report["used_postprocessed_labels"] = True
    else:
        y_hat = y_pred
        report["used_postprocessed_labels"] = False

    # ---- Report + confusion
    rep = binary_report(y_true=y_true, y_hat=y_hat)
    report["report"] = rep

    if make_plots:
        try:
            conf_png = os.path.join(out_dir, "confusion.png")
            plot_confusion(rep["confusion"], save_path=conf_png)
            report.setdefault("artifacts", {})["confusion_plot"] = conf_png
            report["status"]["confusion"] = "ok"
        except Exception as e:
            report["status"]["confusion"] = f"failed: {e}"
    else:
        report["status"]["confusion"] = "skipped"

    # ---- ROC/PR curves (needs prob)
    try:
        fpr, tpr, roc_thr = roc_curve(y_true, prob)
        roc_auc = auc_trapz(fpr, tpr)
        report["roc_auc"] = float(roc_auc)

        if make_plots:
            roc_png = os.path.join(out_dir, "roc.png")
            plot_roc(fpr, tpr, save_path=roc_png, auc=roc_auc)
            report.setdefault("artifacts", {})["roc_plot"] = roc_png
        report["status"]["roc"] = "ok" if make_plots else "skipped"
    except Exception as e:
        report["status"]["roc"] = f"failed: {e}"

    try:
        prec, rec, pr_thr = pr_curve(y_true, prob)
        pr_auc = auc_trapz(rec, prec)
        report["pr_auc"] = float(pr_auc)

        if make_plots:
            pr_png = os.path.join(out_dir, "pr.png")
            plot_pr(rec, prec, save_path=pr_png, auc=pr_auc)
            report.setdefault("artifacts", {})["pr_plot"] = pr_png
        report["status"]["pr"] = "ok" if make_plots else "skipped"
    except Exception as e:
        report["status"]["pr"] = f"failed: {e}"

    report["status"]["predictions"] = "ok"

    _write_report_files(out_dir, report)
    return report


def _write_report_files(out_dir: str, report: Dict[str, Any]) -> None:
    # report.json
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # report.txt (human readable)
    lines = []
    rep = report.get("report", {})
    if rep:
        lines.append(f"acc: {rep.get('acc'):.4f}")
        lines.append(f"precision: {rep.get('precision'):.4f}")
        lines.append(f"recall: {rep.get('recall'):.4f}")
        lines.append(f"f1: {rep.get('f1'):.4f}")
        c = rep.get("confusion")
        if c is not None:
            lines.append(f"confusion: tn={c[0][0]} fp={c[0][1]} fn={c[1][0]} tp={c[1][1]}")
    if "roc_auc" in report:
        lines.append(f"roc_auc: {report['roc_auc']:.4f}")
    if "pr_auc" in report:
        lines.append(f"pr_auc: {report['pr_auc']:.4f}")

    lines.append("")
    lines.append("status:")
    for k, v in (report.get("status") or {}).items():
        lines.append(f"  {k}: {v}")

    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
