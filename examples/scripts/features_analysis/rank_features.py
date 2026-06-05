from __future__ import annotations

"""Rank hand-crafted features for CHB-MIT NPZ segments.

This script is adapted to the **refactored** library and expects you have already
run preprocessing to produce NPZ files under:
  <dataset_dir>/sub-<subject_id>/ses-*/eeg/*<suffix>_*float.npz

Example:
  python examples/scripts/features_analysis/rank_features.py \
    --dataset-dir data/BIDS_CHB-MIT --subject 01 --suffix fd_5s_szx5_prex5
"""

import argparse
import sys
import os
from dataclasses import asdict
from typing import Tuple

import numpy as np
import pandas as pd

from seizure_pred.data.chbmit_npz import CHBMITDataset

# Allow running as a plain script (not a package/module).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from utils import build_feature_specs, extract_features_for_channel  # noqa: E402


def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n1, n2 = len(a), len(b)
    v1 = np.var(a, ddof=1)
    v2 = np.var(b, ddof=1)
    pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / max(1, (n1 + n2 - 2))
    if pooled <= 0:
        return 0.0
    return float((np.mean(a) - np.mean(b)) / np.sqrt(pooled))


def _ttest(a: np.ndarray, b: np.ndarray) -> float:
    try:
        from scipy.stats import ttest_ind
    except Exception as e:
        raise ImportError(
            "This script needs SciPy for t-tests. Install with: pip install seizure-pred[signal]"
        ) from e
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    _, p = ttest_ind(a, b, equal_var=False)
    return float(p)


def _balanced_subset(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    if len(idx0) == 0 or len(idx1) == 0:
        return X, y
    n = min(len(idx0), len(idx1))
    pick = np.concatenate([rng.choice(idx0, size=n, replace=False), rng.choice(idx1, size=n, replace=False)])
    rng.shuffle(pick)
    return X[pick], y[pick]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--subject", "--subject-id", dest="subject", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--use-uint16", action="store_true")
    ap.add_argument("--task", default="prediction", choices=["prediction", "detection"])
    ap.add_argument("--sfreq", type=int, default=128)
    ap.add_argument("--channel", type=int, default=0, help="Which channel index to rank features for")
    ap.add_argument("--max-segments", type=int, default=0, help="0 = use all segments")
    ap.add_argument("--no-mne", action="store_true", help="Disable MNE connectivity features")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(
        "runs", "feature_ranking", f"sub{args.subject}_{args.suffix}_{args.task}"
    )
    os.makedirs(out_dir, exist_ok=True)

    ds = CHBMITDataset(
        dataset_dir=args.dataset_dir,
        subject_id=args.subject,
        suffix=args.suffix,
        use_uint16=args.use_uint16,
        task=args.task,
        print_events=False,
    )

    X = np.asarray(ds.X)
    y = np.asarray(ds.y)

    # Optional cap for quick tests
    if args.max_segments and args.max_segments > 0 and len(X) > args.max_segments:
        X = X[: args.max_segments]
        y = y[: args.max_segments]

    # Balance classes to avoid trivial feature rankings
    X, y = _balanced_subset(X, y)

    specs, feature_names = build_feature_specs(
        sfreq=args.sfreq,
        include_mne_connectivity=not args.no_mne,
        include_scipy_features=True,
        strict=False,
    )

    Xf = extract_features_for_channel(X, args.channel, specs)

    df = pd.DataFrame(Xf, columns=feature_names)
    df["label"] = y
    df.to_csv(os.path.join(out_dir, "features.csv"), index=False)

    # Statistics (t-test + Cohen's d)
    stats_rows = []
    for feat in feature_names:
        g0 = df.loc[df["label"] == 0, feat].to_numpy(dtype=float)
        g1 = df.loc[df["label"] == 1, feat].to_numpy(dtype=float)
        p = _ttest(g0, g1)
        d = _cohen_d(g0, g1)
        stats_rows.append({"feature": feat, "p_value": p, "cohen_d": d})
    stats_df = pd.DataFrame(stats_rows).sort_values("p_value", na_position="last")

    # Model-based importance (RandomForest)
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError("This script requires scikit-learn") from e

    X_clean = df[feature_names].replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.fillna(X_clean.mean(numeric_only=True))
    y_clean = df["label"].astype(int).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X_clean.to_numpy(), y_clean, test_size=0.3, random_state=42, stratify=y_clean
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
    clf.fit(X_train, y_train)
    importances = clf.feature_importances_
    rf_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    # Combine
    report = rf_df.merge(stats_df, on="feature", how="left")
    report.to_csv(os.path.join(out_dir, "feature_ranking.csv"), index=False)

    # Simple plot (top 20)
    try:
        import matplotlib.pyplot as plt
    except Exception:
        plt = None

    if plt is not None:
        top = report.head(20)
        plt.figure(figsize=(10, 8))
        plt.barh(top["feature"][::-1], top["importance"][::-1])
        plt.title("Top feature importances (RandomForest)")
        plt.xlabel("importance")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "top_feature_importance.png"), dpi=200)
        plt.close()

    # Write metadata
    meta = {
        "dataset_dir": args.dataset_dir,
        "subject_id": args.subject,
        "suffix": args.suffix,
        "task": args.task,
        "sfreq": args.sfreq,
        "channel": args.channel,
        "n_segments": int(len(X)),
        "n_features": int(len(feature_names)),
        "feature_specs": [asdict(s) for s in specs],
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        import json

        json.dump(meta, f, indent=2)

    print(f"[rank_features] wrote: {out_dir}")


if __name__ == "__main__":
    main()
