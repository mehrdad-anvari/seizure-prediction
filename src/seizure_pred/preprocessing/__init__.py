"""Preprocessing utilities.

This subpackage is **optional** at install time (requires the ``eeg`` extra,
e.g. ``pip install seizure-pred[eeg]``) because it depends on MNE and EDF
handling.

Important: importing :mod:`seizure_pred` should not fail when optional extras
are not installed. Therefore, we *lazy import* the actual implementations.

Training/inference should rely on :mod:`seizure_pred.data` which loads
already-processed NPZ.
"""

from __future__ import annotations

from typing import Any


def process_chbmit_bids_dataset(*args: Any, **kwargs: Any):
    """Process the CHB-MIT dataset organized as BIDS.

    This is a thin wrapper around :func:`seizure_pred.preprocessing.chbmit_bids.process_chbmit_bids_dataset`
    that avoids importing MNE/EDF dependencies until the function is called.
    """

    try:
        from .chbmit_bids import process_chbmit_bids_dataset as _impl
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Preprocessing utilities require optional dependencies. "
            "Install with: `pip install seizure-pred[eeg]`."
        ) from e
    return _impl(*args, **kwargs)


__all__ = ["process_chbmit_bids_dataset"]
