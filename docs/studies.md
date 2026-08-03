# Studies

This document summarizes the benchmarking studies conducted using the framework.
Each study investigates a single scientific question by varying one or more
configuration parameters while keeping the remaining configuration fixed.

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

| Configuration | Params | FLOPs | Latency (ms) | GPU Memory (MB) | AUC | Sensitivity | FPR/h | Notes |
|---------|-------:|------:|-------------:|----------------:|----:|------------:|------:|------|
| | | | | | | | | |
| | | | | | | | | |
| | | | | | | | | |

---

## STUDY-002: <Study Title>

**Question**

> What is being evaluated?

**Base Configuration**

`configs/studies/<config_name>.yaml`

**Variables**

| Parameter | Values |
|----------|--------|
| | |

### Results

| Configuration | Params | FLOPs | Latency (ms) | GPU Memory (MB) | AUC | Sensitivity | FPR/h | Notes |
|---------|-------:|------:|-------------:|----------------:|----:|------------:|------:|------|
| | | | | | | | | |
| | | | | | | | | |