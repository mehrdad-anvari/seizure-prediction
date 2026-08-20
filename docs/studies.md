# Studies

This document summarizes the benchmarking studies conducted using the framework.
Each study investigates a single scientific question by varying one or more
configuration parameters while keeping the remaining configuration fixed.

For all studies we used these **Processing Options** unless otherwise mentioned:

`save_uint16: False`
`apply_filter: True`
`filter_type: IRR`
`l_freq: 0.5`
`h_freq: 50.0`
`apply_ica: False`
`apply_downsampling: True`
`downsample_method: polyphase`
`sfreq_new: 128.0`
`normalize: instance_norm`
`segment_sec: 5`
`preictal_oversample_factor: 1`
`seizure_oversample_factor: 1`
`preictal_minutes: 15`
`post_buffer_minutes: 60`
`pre_buffer_minutes: 45`

Unless otherwise stated, normalization is applied as an offline training transform after loading the unnormalized segments. Each channel is normalized independently within each segment over its time samples.

---

## STUDY-001: Data Splitting Strategy

### Question

> Which inner-fold data splitting strategy provides the best generalization performance for seizure prediction?

### Motivation

The outer folds are generated using a leave-one-preictal-out strategy without shuffling either the preictal or interictal segments. This prevents data leakage caused by the strong temporal correlation between adjacent EEG windows. Preserving the chronological order of the test set is also essential because the moving-average post-processing stage operates on consecutive predictions.

Since moving-average post-processing is not applied to the training or validation sets, the temporal order of samples within the inner folds is not required. This allows us to investigate whether alternative splitting strategies can improve model generalization.

### Hypothesis

Randomized or stratified sampling may expose the model to a more diverse training set, leading to improved predictive performance (e.g., AUC, sensitivity, and FPR/h). Conversely, preserving the temporal order may encourage the model to exploit correlations between adjacent windows, which could artificially inflate validation performance while reducing its ability to generalize to unseen seizure events.

### Base Configuration

`configs/studies/study001.yaml`


### Configurations

| Configuration | Modified Parameters |
|--------------|---------------------|
| `EXP-001-A` | `inner_method=KFold`, `inner_n_fold=5`, `inner_shuffle=false`, `inner_mode=per_event_strata`, `inner_M=10` |
| `EXP-001-B` | `inner_method=KFold`, `inner_n_fold=5`, `inner_shuffle=false`, `inner_mode=split` |
| `EXP-001-C` | `inner_method=KFold`, `inner_n_fold=5`, `inner_shuffle=true`, `inner_mode=split` |
| `EXP-001-D` | `inner_method=LOO`, `inner_n_fold=7`, `inner_shuffle=false`, `inner_mode=split` |

### Results
`Threshold = 0.5` 
`No moving average`
`No calibration`

`TPR = Sensivity`
`FPR/h suppressed: Ignore positive prediction for 5 mins after a detection`

| Configuration | AUC    | F1     | TPR  | FPR/h  | FPR/h supp. | 
|---------------|:------:|:------:|:----:|:------:|:------------:|
| `EXP-001-A`   | 0.8648 | 0.3228 | 1.0  | 110.01 | 5.69 |
| `EXP-001-B`   | 0.7625 | 0.2870 | 1.0  |  96.32 | 5.92 |
| `EXP-001-C`   |        |        |      |        |      |
| `EXP-001-D`   |        |        |      |        |      |
---

## STUDY-002: Signal Normalization

### Question

> Which signal normalization method provides the best generalization performance for seizure prediction?

### Motivation

Normalization can reduce segment- and channel-specific amplitude differences and make the input distribution easier for the model to learn. This study compares no normalization with z-score and robust normalization as offline training transforms while keeping preprocessing, data splitting, model, optimization, and training settings fixed.

### Hypothesis

Z-score normalization is expected to provide a strong baseline by centering each signal and scaling it by its standard deviation. Robust normalization may perform better when EEG contains substantial artifacts or outliers because it uses the median and median absolute deviation. Removing normalization may retain useful amplitude information but can make optimization and cross-subject generalization more difficult.

### Base Configuration

`configs/studies/study002.yaml`

### Configurations

| Configuration | Normalization | Offline Transform | Data Suffix |
|--------------|---------------|-------------------|-------------|
| `EXP-002-A` | None | `offline_transforms: []` | `_fd_5s` |
| `EXP-002-B` | Z-score | `offline_transforms: ["instance_norm"]` | `_fd_5s` |
| `EXP-002-C` | Robust | `offline_transforms: ["robust_norm"]` | `_fd_5s` |

All experiments load the same unnormalized `_fd_5s` segments. Offline transforms are applied once when the training dataset is loaded and operate independently on every channel within every segment. `instance_norm` centers by the channel mean and scales by its standard deviation; `robust_norm` centers by the channel median and scales by the median absolute deviation estimate.

### Results

| Configuration | Params | FLOPs | Latency (ms) | GPU Memory (MB) | AUC | Sensitivity | FPR/h | Notes |
|---------|-------:|------:|-------------:|----------------:|----:|------------:|------:|------|
