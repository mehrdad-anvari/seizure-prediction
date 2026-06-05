"""Compute & visualize feature transforms on an EEG sample.

Updated for the new package layout: `seizure_pred.transforms`.

This script is intentionally light and works with a user-provided EEG array.

Input formats:
  - .npy containing an array shaped (channels, time) or (batch, channels, time)
  - .npz containing key `eeg`

Example:
  python examples/scripts/visualize_features.py --input sample.npy \
    --sfreq 256 --out features.png

Optional:
  - If SciPy is installed, you can enable filterbank / bandpower style transforms.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt


def _require_scipy() -> None:
    try:
        import scipy  # noqa: F401
    except Exception as e:
        raise ImportError(
            "This demo requires SciPy. Install with: pip install seizure-pred[signal] (or pip install scipy)"
        ) from e


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize feature transforms")
    p.add_argument("--input", required=True, help="Path to .npy or .npz (key 'eeg')")
    p.add_argument("--sfreq", type=float, default=256.0, help="Sampling frequency (Hz)")
    p.add_argument("--out", default=None, help="Output image path (default: next to input)")
    p.add_argument(
        "--mode",
        choices=["basic_stats", "band_power"],
        default="basic_stats",
        help="Which feature demo to run",
    )
    return p.parse_args()


def _load_eeg(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        eeg = np.load(path)
    elif path.suffix.lower() == ".npz":
        d = np.load(path)
        if "eeg" not in d:
            raise KeyError(".npz must contain key 'eeg'")
        eeg = d["eeg"]
    else:
        raise ValueError("Input must be .npy or .npz")

    # Accept (B,C,T) or (C,T)
    if eeg.ndim == 3:
        eeg = eeg[0]
    if eeg.ndim != 2:
        raise ValueError(f"Expected (C,T) or (B,C,T), got shape={eeg.shape}")
    return eeg.astype(np.float32)


def main() -> None:
    args = _parse_args()
    path = Path(args.input)
    eeg = _load_eeg(path)

    if args.mode == "basic_stats":
        # The old repo exposes many single-stat transforms; here we demo LineLength.
        from seizure_pred.transforms.feature.basic_stats import LineLength

        tr = LineLength()
        feat = tr(eeg)
        title = "Basic stats (LineLength)"

    elif args.mode == "band_power":
        # SciPy required (filters/psd in most implementations)
        _require_scipy()
        from seizure_pred.transforms.feature.band_power import BandPowerTransform

        tr = BandPowerTransform(sfreq=args.sfreq)
        feat = tr(eeg)
        title = "Band power features"
    else:
        raise AssertionError("unreachable")

    # Normalize for display: (channels, features) preferred
    feat = np.asarray(feat)
    plt.figure()
    plt.imshow(feat, aspect="auto")
    plt.title(title)
    plt.xlabel("feature")
    plt.ylabel("channel")

    out = Path(args.out) if args.out else path.with_suffix(".features.png")
    plt.savefig(out, bbox_inches="tight", dpi=150)
    print(f"[visualize_features] wrote {out}")


if __name__ == "__main__":
    main()
