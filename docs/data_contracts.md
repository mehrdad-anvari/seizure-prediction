# Data Contracts

This library standardizes dataset and dataloader outputs so new components can be added without modifying the core training loops or evaluation logic.

## Dataset `__getitem__` Contract

Any custom dataset registered under the `DATASETS` registry **must** return a 3-tuple containing:

1. **`x`**: `torch.Tensor` representing the input features (typically with shape `(C, T)` where `C` is the number of EEG channels and `T` is the number of time steps).
2. **`y`**: `torch.Tensor` representing the targets (typically a scalar class index 0/1 for binary classification, or a multi-label vector).
3. **`meta`**: `dict` containing metadata about the sample (e.g., event ID, start time, subject ID, raw label).

## Instance Dataloader Contract

An instance dataloader yields batches where:

- **`x`**: `torch.Tensor` with shape `(B, C, T)` (where `B` is batch size).
- **`y`**: `torch.Tensor` with shape `(B,)` (or `(B, K)` for multi-class/multi-label tasks).
- **`meta`**: List of dictionaries of length `B` containing metadata for each batch item.

## MIL (Multiple Instance Learning) Dataloader Contract

Multiple Instance Learning (MIL) loaders yield bags of instances where:

- **`x`**: `torch.Tensor` with shape `(B, bag_size, C, T)`.
- **`y`**: `torch.Tensor` with shape `(B,)`.
- **`meta`**: List of list of dictionaries of length `B` (each inner list has length `bag_size`).

## CHB-MIT Dataset Labeling Conventions

Depending on the task, the labels are interpreted as follows:

### Prediction Task
- **Positive class (1)**: `preictal` (the period leading up to a seizure).
- **Negative class (0)**: `interictal` (the period far from any seizures).

### Detection Task
- **Positive class (1)**: `seizure` (the active seizure period).
- **Negative class (0)**: `interictal`, `preictal`, `pre_buffer`, `post_buffer`.
