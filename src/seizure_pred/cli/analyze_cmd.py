from __future__ import annotations

import argparse

from seizure_pred.analysis.runner import analyze_run


def add_analyze_cmd(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("analyze", help="Analyze a run and write plots/reports")
    p.add_argument("--run-dir", required=True, help="Run directory containing predictions/history")
    p.add_argument("--out-dir", default=None, help="Output directory (default: <run-dir>/analysis)")
    p.add_argument("--threshold", type=float, default=0.5, help="Threshold for y_pred if needed")
    p.add_argument("--prefer-postprocessed", action="store_true", help="Use y_pred_post if present")
    p.add_argument("--no-plots", action="store_true", help="Skip writing plots (CI-friendly)")
    p.set_defaults(func=run_analyze_cmd)


def run_analyze_cmd(args: argparse.Namespace) -> None:
    analyze_run(
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        threshold=args.threshold,
        prefer_postprocessed=args.prefer_postprocessed,
        make_plots=not args.no_plots,
    )
