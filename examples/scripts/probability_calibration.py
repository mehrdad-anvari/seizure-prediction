"""Calibrate / choose threshold for run probabilities.

Updated for the new run artifacts layout (`predictions.jsonl`).

This script intentionally avoids hard dependencies. If scikit-learn is
installed, it can fit Platt scaling. Otherwise it will fall back to selecting
the best threshold by F1 score.

Outputs a JSON file containing:
  - best_threshold_f1
  - optional platt_scale parameters

Example:
  python examples/scripts/probability_calibration.py \
    --run-dir runs/<...>/split_0 \
    --out calibration.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from seizure_pred.analysis.runs import load_predictions


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrate probabilities / choose threshold")
    p.add_argument("--run-dir", required=True, help="Run directory containing predictions.jsonl")
    p.add_argument("--out", default=None, help="Output JSON path (default: <run_dir>/analysis/calibration.json)")
    p.add_argument("--prefer-postprocessed", action="store_true", help="Use y_pred_post if present")
    return p.parse_args()


def _best_threshold_f1(y_true: np.ndarray, prob: np.ndarray, *, n: int = 2001) -> float:
    # Threshold grid in [0,1]
    thr = np.linspace(0.0, 1.0, n)
    best_t = 0.5
    best_f1 = -1.0
    # vectorized confusion computations
    y_true_b = y_true.astype(np.int64)
    for t in thr:
        y_hat = (prob >= t).astype(np.int64)
        tp = int(((y_hat == 1) & (y_true_b == 1)).sum())
        fp = int(((y_hat == 1) & (y_true_b == 0)).sum())
        fn = int(((y_hat == 0) & (y_true_b == 1)).sum())
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _try_platt_scale(prob: np.ndarray, y_true: np.ndarray) -> Optional[Dict[str, float]]:
    """Fit Platt scaling if sklearn is available.

    Returns dict with keys {"a", "b"} where calibrated = sigmoid(a*logit(p) + b).
    """
    try:
        from sklearn.linear_model import LogisticRegression  # type: ignore
    except Exception:
        return None

    eps = 1e-6
    p = np.clip(prob, eps, 1 - eps)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    y = y_true.astype(int)

    clf = LogisticRegression(solver="lbfgs")
    clf.fit(logit, y)
    a = float(clf.coef_.ravel()[0])
    b = float(clf.intercept_.ravel()[0])
    return {"a": a, "b": b}


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir)
    preds_path = run_dir / "predictions.jsonl"
    if not preds_path.exists():
        raise SystemExit(f"Missing predictions.jsonl: {preds_path}")

    y_true, prob, y_pred, y_pred_post = load_predictions(str(preds_path))
    if y_true.size == 0:
        raise SystemExit("Empty predictions.jsonl")

    best_t = _best_threshold_f1(y_true, prob)
    platt = _try_platt_scale(prob, y_true)

    out_path = Path(args.out) if args.out else (run_dir / "analysis" / "calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "best_threshold_f1": float(best_t),
        "platt_scaling": platt,
        "note": "Install scikit-learn to enable Platt scaling; otherwise only threshold selection is computed.",
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[calibration] wrote {out_path}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
