# Cross-validation & splits

Splitters live in `seizure_pred.data.splits`. They operate on any dataset
exposing `y`, `group_ids`, `metadata`, and (optionally) `is_used_in_train` and
`base_indices`. `SubsetWithInfo` preserves class/group/base-index info across
nested splits, enabling leakage-aware evaluation.

## When training only uses "usable" samples

Datasets expose `is_used_in_train` (a boolean mask). Prediction tasks train on
`{preictal, interictal}`; detection trains on
`{seizure, interictal, preictal, post_buffer, pre_buffer}`. Buffers are
excluded from training but kept for evaluation. Splitters filter to
`is_used_in_train` automatically.

## Leave-one-preictal-out (`LOO`)

`leave_one_out(dataset, method=..., shuffle_interictal=..., random_state=...)`

Holds out one positive (seizure/preictal) event per fold and assigns interictal
windows to folds. `method`:

| Method | Behaviour |
|--------|-----------|
| `balanced` | interictal windows partitioned evenly in chronological order |
| `balanced_shuffled` | balanced + randomised interictal selection |
| `nearest` | test interictal windows are the **temporally nearest** to the held-out event (reduces train/test distribution drift) |

`leave_one_preictal(...)` is an alias; `shuffle_interictal=True` implies
`balanced_shuffled`.

## Stratified K-Fold (`stratified`)

`stratified_kfold(dataset, n_folds=5, shuffle=..., random_state=...)`

scikit-learn `StratifiedKFold` — the default inner CV. Keeps label distribution
roughly constant across folds.

## Custom chronological K-Fold (`KFold`)

`KFold(dataset, n_fold=5, shuffle=..., random_state=..., mode=..., M=10)`

| Mode | Behaviour |
|------|-----------|
| `random_split` | shuffle within class then split |
| `split` | chronological split within class |
| `strata` | per-class strata of `M` samples per fold (limits overlap leakage) |
| `per_event_strata` | stratify *within each event* then combine (default) |

`split_into_strata(indices, N, M)` assigns `M` samples per stratum per pass and
validates `M ≤ ceil(len/(N+1))` to prevent leakage from overlapping windows.

## Dispatch helper

`make_cv_splitter(dataset, *, mode=None, method=None, n_folds=None, n_fold=None,
shuffle=False, random_state=0, M=10)`

- `mode`/`method` `LOO` / `leave_one_preictal` / `leave_one_out` → leave-one-out.
  For LOO, `mode` may be `balanced` / `balanced_shuffled` / `nearest` to choose
  the interictal strategy.
- `stratified` → scikit-learn StratifiedKFold.
- `KFold` → custom chronological K-Fold (`mode` selects the strata mode).

## Nested CV in the CLI

Add a `cv:` block to run nested CV. The outer loop evaluates on held-out
events; the inner loop selects the best epoch and produces per-inner-fold
validation probabilities used for **calibration** and **AUC-weight ensembling**.

```yaml
cv:
  outer_method: "LOO"            # or "KFold"
  outer_mode: "nearest"          # LOO interictal strategy, or a KFold mode
  outer_n_fold: 5
  inner_method: "KFold"
  inner_mode: "per_event_strata"
  inner_n_fold: 5
  inner_M: 10
  random_state: 42
```
