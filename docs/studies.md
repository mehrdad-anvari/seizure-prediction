# Studies

This document summarizes the benchmarking studies conducted using the framework.
Each study investigates a single scientific question by varying one or more
configuration parameters while keeping the remaining configuration fixed.

For all studies we used these **Processing Options** unless otherwise mentioned:

`save_uint16: False`
`apply_filter: True`
`filter_type: FIR`
`l_freq: 0.5`
`h_freq: 50.0`
`apply_ica: False`
`apply_downsampling: True`
`downsample_method: polyphase`
`sfreq_new: 128.0`
`normalize: zscore`
`segment_sec: 5`
`preictal_oversample_factor: 1`
`seizure_oversample_factor: 1`
`preictal_minutes: 15`
`post_buffer_minutes: 60`
`pre_buffer_minutes: 45`

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
| `EXP-001-A`   | 0.8353 | 0.2904 | 1.0  | 109.03 | 6.70 | 
| `EXP-001-B`   | 0.7905 | 0.2371 | 1.0  | 113.55 | 6.97 | 
| `EXP-001-C`   | 0.7819 | 0.2324 | 1.0  | 120.92 | 7.27 | 
| `EXP-001-D`   | 0.8024 | 0.2630 | 1.0  | 116.58 | 6.88 | 
---

## STUDY-002: Signal Normalization

### Question

> Which signal normalization method provides the best generalization performance for seizure prediction?

### Motivation

Normalization can reduce subject- and channel-specific amplitude differences and make the input distribution easier for the model to learn. This study compares no normalization with z-score and robust normalization during preprocessing while keeping the data splitting, model, optimization, and training settings fixed.

### Hypothesis

Z-score normalization is expected to provide a strong baseline by centering each signal and scaling it by its standard deviation. Robust normalization may perform better when EEG contains substantial artifacts or outliers because it uses the median and interquartile range. Removing normalization may retain useful amplitude information but can make optimization and cross-subject generalization more difficult.

### Base Configuration

`configs/studies/study002.yaml`

### Configurations

| Configuration | Normalization | Preprocessing Option | Data Suffix |
|--------------|---------------|----------------------|-------------|
| `EXP-002-A` | None | omit `--normalize` | `_fd_5s` |
| `EXP-002-B` | Z-score | `--normalize zscore` | `_fdn_5s` |
| `EXP-002-C` | Robust | `--normalize robust` | `_fdn_5s` |

Normalization is applied channel-wise to each continuous recording before segmentation. The training pipeline does not apply additional normalization.

Both normalized variants currently use the same `_fdn_5s` suffix. Preprocessing robust data in the same dataset directory therefore overwrites the z-score NPZ files and processing-options files. Preserve each normalized dataset in a separate dataset directory, or complete the corresponding run before preprocessing the next normalized variant.

### Results

| Configuration | Params | FLOPs | Latency (ms) | GPU Memory (MB) | AUC | Sensitivity | FPR/h | Notes |
|---------|-------:|------:|-------------:|----------------:|----:|------------:|------:|------|
