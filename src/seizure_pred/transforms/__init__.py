"""EEG transforms.

This subpackage contains signal- and feature-level transforms ported from the
original repository.

Transforms are plain Python classes (callables). For convenience, we expose a
small dependency-safe factory:

  from seizure_pred.transforms import create_transform
  tr = create_transform("band_power", sfreq=256)
  features = tr(eeg)  # eeg shape (C,T)
"""

from .base import BaseTransform  # noqa: F401
from .registry import create_transform, list_transforms, TransformSpec  # noqa: F401
