"""Analysis utilities for seizure_pred runs.

This package provides:
- loaders for standardized artifacts (predictions.jsonl, history.jsonl)
- metrics and plotting helpers
- an end-to-end runner used by `seizure-pred analyze`
"""

from .runner import analyze_run  # noqa: F401
from .runs import load_predictions  # noqa: F401
from .metrics import binary_report, roc_curve, pr_curve, auc_trapz, moving_average_segmented, clinical_metrics  # noqa: F401
from .summary import analyze_multi_split_summary  # noqa: F401
from .nested_predictions import (  # noqa: F401
    analyze_interictal_prob,
    analyze_pp_scatter_combined,
    analyze_preictal_prob,
    ananlyze_preictal_prob,
)
from .plots import (  # noqa: F401
    plot_confusion,
    plot_history,
    plot_interictal_combined,
    plot_interictal_prob,
    plot_pr,
    plot_preictal_prob,
    plot_prob_vs_pp_scatter_combined,
    plot_roc,
)

__all__ = [
    "analyze_run",
    "load_predictions",
    "binary_report",
    "roc_curve",
    "pr_curve",
    "auc_trapz",
    "moving_average_segmented",
    "clinical_metrics",
    "analyze_multi_split_summary",
    "analyze_interictal_prob",
    "analyze_preictal_prob",
    "analyze_pp_scatter_combined",
    "ananlyze_preictal_prob",
    "plot_history",
    "plot_confusion",
    "plot_roc",
    "plot_pr",
    "plot_interictal_combined",
    "plot_interictal_prob",
    "plot_preictal_prob",
    "plot_prob_vs_pp_scatter_combined",
]
