# Legacy pipeline (main repo) mapped to the new library

This project was refactored into a reusable Python package. The goal of this page is to **recreate the same end‑to‑end workflow** you used in `seizure-prediction-main.zip`:

1) preprocess CHB‑MIT BIDS data into NPZ sessions
2) run **nested cross‑validation** (outer + inner folds) per subject

The new repo uses a configuration-driven trainer, but you can still run a "legacy style" pipeline using the example script below.

---

## 1) Preprocess (old: `python dataset/preprocess.py ...`)

**Old (main repo)**

```bash
python dataset/preprocess.py --apply_filter --subjects 1 2 3 ...
```

**New (this repo)**

```bash
seizure-pred preprocess-chbmit \
  --dataset-dir /path/to/BIDS_CHB-MIT \
  --subject 1,2,3 \
  --save-uint16 \
  --filter-type FIR --l-freq 0.5 --h-freq 50 \
  --sfreq-new 128 --downsample-method polyphase \
  --normalize zscore
```

Notes:
- `--no-filter`, `--no-ica`, `--no-downsample` disable steps.
- Output NPZ sessions are written under each subject folder, e.g. `sub-01/ses-*/eeg/*_float.npz` and/or `*_uint16.npz`.

---

## 2) Train with nested CV (old: `train.py` outer+inner)

**Old (main repo)** had arguments like:

- `--apply_normalization`
- `--use_uint16`
- `--subject_id 1`
- `--model eegwavenet`
- `--batch_size 32`
- `--outer_cv_mode leave_one_preictal`
- `--outer_cv_method balanced`
- `--inner_cv_mode stratified`
- `--inner_cv_method balanced`
- `--n_fold 5`

**New (this repo):** use the example script:

```bash
python examples/scripts/legacy_nested_cv.py \
  --dataset-dir /path/to/BIDS_CHB-MIT \
  --subject-id 1 \
  --suffix fd_5s_szx5_prex5 \
  --task prediction \
  --model eegwavenet \
  --batch-size 32 \
  --epochs 50 \
  --lr 1e-3 \
  --use-uint16 \
  --apply-normalization \
  --outer-cv-mode leave_one_preictal --outer-cv-method balanced \
  --inner-cv-mode stratified --inner-cv-method balanced \
  --n-fold 5
```

Outputs:
- `runs/legacy_nested_cv/<timestamp>/sub-XX/outer_k/inner_j/` contains:
  - `logs/training.log`
  - `artifacts/` (metrics, history, checkpoints, predictions)

---

## 3) Logging

The new trainer now logs to both console and a file **per split/run directory**, similar to `train.py` in the old repo:

- `.../logs/training.log`

---

If you prefer to run training purely via the config-driven CLI (`seizure-pred train --config ...`) you still can; the script here is just the closest equivalent to the nested-CV workflow from the old code.
