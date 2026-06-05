"""Built-in dataloader plugins.

We keep registration lightweight: importing this package registers the canonical
loaders via decorator side-effects in the modules below.

Legacy ``*_loader.py`` modules are still shipped for backwards compatibility,
but they are **not** imported here to avoid duplicate registry entries.
"""

# Canonical loaders (decorator-based registration on import)
from . import torch_default as _torch_default  # noqa: F401
from . import undersample as _undersample  # noqa: F401
from . import mil as _mil  # noqa: F401

__all__ = []
