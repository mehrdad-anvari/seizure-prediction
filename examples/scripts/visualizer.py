"""Quick visualization helpers for new run artifacts.

This is a rewritten version of the legacy `visualizer.py`.

Currently supports:
  - Plot probability timeline from `predictions.jsonl`
  - Plot training history from `history.jsonl` (if present)

Example:
  python examples/scripts/visualizer.py --run-dir runs/<...>/split_0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from seizure_pred.analysis.runs import load_predictions
from seizure_pred.analysis.plots import plot_history


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize seizure_pred artifacts")
    p.add_argument("--run-dir", required=True, help="Run directory")
    p.add_argument("--show", action="store_true", help="Show interactive windows instead of saving")
    p.add_argument("--out-dir", default=None, help="Output directory (default: <run_dir>/analysis)")
    return p.parse_args()


def _plot_prob_timeline(pred_path: Path, save_path: Path | None) -> None:
    y_true, prob, y_pred, y_pred_post = load_predictions(str(pred_path))
    if y_true.size == 0:
        raise SystemExit("Empty predictions.jsonl")

    plt.figure()
    plt.plot(prob)
    plt.title("Predicted probability over samples")
    plt.xlabel("sample")
    plt.ylabel("p(seizure)")
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = run_dir / "predictions.jsonl"
    if preds.exists():
        out_png = None if args.show else (out_dir / "prob_timeline.png")
        _plot_prob_timeline(preds, out_png)
        print(f"[visualizer] prob timeline -> {out_png or '<shown>'}")
    else:
        print(f"[visualizer] missing: {preds}")

    hist = run_dir / "history.jsonl"
    if hist.exists():
        out_png = None if args.show else (out_dir / "history.png")
        if out_png is None:
            plot_history(str(hist), save_path=None)
        else:
            plot_history(str(hist), save_path=str(out_png))
        print(f"[visualizer] history -> {out_png or '<shown>'}")
    else:
        print(f"[visualizer] missing: {hist}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
