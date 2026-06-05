"""Analysis utilities for seizure_pred runs.

This package provides:
- loaders for standardized artifacts (predictions.jsonl, history.jsonl)
- metrics and plotting helpers
- an end-to-end runner used by `seizure-pred analyze`
"""

from .runner import analyze_run  # noqa: F401
from .runs import load_predictions  # noqa: F401
from .metrics import binary_report, roc_curve, pr_curve, auc_trapz  # noqa: F401
from .plots import plot_history, plot_confusion, plot_roc, plot_pr  # noqa: F401

__all__ = [
    "analyze_run",
    "load_predictions",
    "binary_report",
    "roc_curve",
    "pr_curve",
    "auc_trapz",
    "plot_history",
    "plot_confusion",
    "plot_roc",
    "plot_pr",
]
