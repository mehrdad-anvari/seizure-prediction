# Release checklist

This checklist is meant to help you cut a clean release (e.g., `v0.2.0`) with reproducible artifacts.

## 1) Sanity checks (local)
- [ ] `python -m pip install -U pip`
- [ ] `pip install -e ".[train,viz]"` (and `.[eeg]` if you want preprocessing)
- [ ] `pytest -q`
- [ ] `PYTHONPATH=src seizure-pred --help`
- [ ] `PYTHONPATH=src seizure-pred list`
- [ ] Train a tiny synthetic run:
  - [ ] `PYTHONPATH=src seizure-pred train --config examples/config_prediction.yaml --split-index 0`
- [ ] Predict + analyze on the produced run dir:
  - [ ] `PYTHONPATH=src seizure-pred predict --config examples/config_prediction.yaml --checkpoint <best.pt> --split-index 0 --out-dir <out>`
  - [ ] `PYTHONPATH=src seizure-pred analyze --run-dir <out> --no-plots`

## 2) Versioning
- [ ] Ensure `pyproject.toml` version is correct.
- [ ] Update `CHANGELOG.md` (move items from Unreleased into the version section).
- [ ] Commit with message like: `Release 2025-12-20 vX.Y.Z`.

## 3) Git tag
- [ ] `git tag -a vX.Y.Z -m "vX.Y.Z"`
- [ ] `git push --tags`

## 4) Build & verify wheel/sdist (optional but recommended)
- [ ] `pip install build twine`
- [ ] `python -m build`
- [ ] `twine check dist/*`

## 5) Publish (optional)
- [ ] Upload to PyPI (or internal index) with twine.
- [ ] Create a GitHub Release and attach release notes (from `CHANGELOG.md`).

## Suggested next versions
- `0.2.1`: bugfix only
- `0.3.0`: add a new public capability (new evaluator/splitter/postprocess defaults, etc.)
