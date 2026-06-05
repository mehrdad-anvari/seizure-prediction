# Transforms

This package provides lightweight, dependency-safe transforms that can be used in
data pipelines and feature extraction.

## Structure

- `signal/`: signal-level transforms (normalization, filterbanks, rearrangements)
- `feature/`: feature extraction transforms
- `registry.py`: a small factory that can lazily construct transforms by name

## Optional dependencies

Some transforms require optional dependencies:

- Install SciPy-backed transforms:
  - `pip install -e ".[signal]"`
- Install EEG connectivity transforms:
  - `pip install -e ".[eeg]"`

If an optional dependency is missing, the module will not crash the library at
import time; instead you will get a clear `ImportError` only when constructing or
running the affected transform.
