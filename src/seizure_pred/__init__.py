"""Seizure prediction & detection library.

This package provides:
- Dataset preprocessing (e.g., CHB-MIT in BIDS)
- Dataset loaders for prediction and detection tasks
- Training loops with plug-in registries (loss/optimizer/scheduler/model/dataset)
- Inference utilities for offline/online prediction

The API is intentionally modular: you can register your own components without
editing the core training loop.
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.2.0"
