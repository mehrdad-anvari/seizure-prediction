# Config validation

The CLI validates config files before running training/experiments.

- Unknown keys are rejected.
- Types are checked (int/float/bool/str/list/dict and nested config blocks).
- Optional fields can be null.

If validation fails, you'll get a readable list of errors with full paths.
