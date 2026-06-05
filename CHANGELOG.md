# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]
### Added
- (Reserved)

### Changed
- (Reserved)

### Fixed
- (Reserved)

## [0.2.0] - 2025-12-20
### Added
- Converted repository into a pip-installable library (src/ layout) with a stable public API surface.
- Registry/factory plugin system for datasets, dataloaders, models, losses, optimizers, schedulers, evaluators, callbacks, and post-processing.
- Trainer and MIL Trainer with standardized run artifacts (config/history/metrics/predictions) and schema versioning.
- Separate inference API and `seizure-pred predict` CLI.
- `seizure-pred analyze` CLI to generate reports and plots from run artifacts.
- CHB-MIT preprocessing pipeline (BIDS → NPZ) behind optional dependency extra `.[eeg]`.
- CI smoke test plumbing and templates to add new plugins with minimal conflicts.
