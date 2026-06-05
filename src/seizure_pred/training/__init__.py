from __future__ import annotations

# Re-export registries
from .registries import (  # noqa: F401
    DATASETS,
    DATALOADERS,
    MODELS,
    LOSSES,
    OPTIMIZERS,
    SCHEDULERS,
    EVALUATORS,
    CALLBACKS,
    POSTPROCESSORS,
    list_all,
)


def register_all() -> None:
    """Import subpackages to register built-in plugins.

    This is intentionally *not* executed at import time to avoid pulling in
    torch-heavy modules when users only want lightweight commands (e.g. analyze).
    """

    # Force plugin registration (import side-effects are intentional here)
    from . import datasets as _datasets  # noqa: F401
    from . import dataloaders as _dataloaders  # noqa: F401
    from . import components as _components  # noqa: F401
    from . import evaluators as _evaluators  # noqa: F401
    from . import callbacks as _callbacks  # noqa: F401
    from . import postprocess as _postprocess  # noqa: F401
