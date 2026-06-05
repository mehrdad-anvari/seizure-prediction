"""Analyze a run directory produced by the new training pipeline.

This is the *updated* version of the old analysis script, rewritten
to use the new library APIs.

It expects a `run_dir` that contains (at minimum):
  - predictions.jsonl

Optionally (if present) it will also use:
  - history.jsonl
  - config.json

Artifacts are written to `<run_dir>/analysis/`.

Example:
  python examples/scripts/analyze_results.py --run-dir runs/my_run/.../split_0

CLI alternative:
  seizure-pred analyze --run-dir runs/my_run/.../split_0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from seizure_pred.analysis.runner import analyze_run


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze a seizure_pred run directory")
    p.add_argument("--run-dir", required=True, help="Path to run directory (contains predictions.jsonl)")
    p.add_argument("--out-dir", default=None, help="Output directory (default: <run_dir>/analysis)")
    p.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for metrics")
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip writing plots (still writes report.json/report.txt)",
    )
    p.add_argument(
        "--prefer-raw-labels",
        action="store_true",
        help="Prefer raw y_pred over postprocessed y_pred_post if available",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    report = analyze_run(
        args.run_dir,
        out_dir=args.out_dir,
        threshold=args.threshold,
        prefer_postprocessed=not args.prefer_raw_labels,
        make_plots=not args.no_plots,
    )

    # Print short summary for convenience
    out_dir = Path(report.get("out_dir") or Path(args.run_dir) / "analysis")
    print(json.dumps({
        "out_dir": str(out_dir),
        "status": report.get("status", {}),
        "acc": (report.get("report") or {}).get("acc"),
        "f1": (report.get("report") or {}).get("f1"),
        "roc_auc": report.get("roc_auc"),
        "pr_auc": report.get("pr_auc"),
    }, indent=2))


if __name__ == "__main__":
    main()
