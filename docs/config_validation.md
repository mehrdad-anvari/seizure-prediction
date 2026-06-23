# Configuration Validation

To catch configuration errors early, the library's command-line interface (CLI) validates all YAML/JSON config files before launching any training or experiment tasks.

## Validation Rules

- **Unknown Keys**: Any keys not defined in the configuration schema are strictly rejected.
- **Type Safety**: Field values are checked against their defined types (including primitive types like `int`, `float`, `bool`, `str`, collection types like `list` and `dict`, and nested configuration groups).
- **Optional Fields**: Optional configuration parameters can explicitly be set to `null` (None).

## Failure Handling

If a configuration file fails validation, the CLI prints a readable list of specific errors with full dot-notation paths showing exactly where the errors occurred (e.g. `train.data.batch_size: expected int, got str`).

Validation is implemented in: [validate.py](file:///e:/Projects/seizure/library/seizure-prediction/src/seizure_pred/core/validate.py).
