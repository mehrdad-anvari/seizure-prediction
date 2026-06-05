"""Optional dependency helpers.

The library includes some models that rely on heavier / platform-specific
dependencies (e.g., torch-geometric). We keep those dependencies optional to
avoid making the base installation brittle.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Optional


def is_torch_geometric_available() -> bool:
    """Return True if torch_geometric can be imported."""
    return find_spec("torch_geometric") is not None


def require_torch_geometric(err: Optional[BaseException] = None) -> None:
    """Raise an ImportError with a friendly install hint."""
    msg = (
        "This feature requires 'torch-geometric'. "
        "Install optional GNN dependencies with: pip install seizure-pred[gnn] "
        "(you may also need the matching PyG wheels for your torch/CUDA setup)."
    )
    raise ImportError(msg) from err
