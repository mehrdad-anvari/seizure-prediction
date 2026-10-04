# Pipeline audit: why "shuffle" runs score low (2026-10-03 / 04)

Branch: `fix/pipeline-audit-shuffle-perf` (from `study/eegwavenet`).
Question: run002 (`configs/full_dataset/run002.yaml`, CE-stSENet, nested CV with
`outer_shuffle: true`) gives window F1 ≈ 0.3 although papers report 80–90 %. Is the
pipeline broken?

**Short answer: no.** The low F1 is not a bug. The shuffle mode never created
the leakage that inflates published numbers, F1 is computed at ~3 % preictal
prevalence, and the real bottleneck is recall on an unseen preictal event. When the
leakage the papers have is reproduced (random window split), the same pipeline gives
AUC 0.9997 / F1 0.92 on sub-11.

All numbers below: calibration `none`, MA window 1, threshold 0.5 unless stated.
"Balanced" metrics re-weight interictal windows so both classes have equal total
weight (≈ evaluation at 50 % prevalence, as most papers report).

## 1. Diagnosis of the existing run002 results (11 subjects)

| | no shuffle | shuffle |
|---|---|---|
| mean window AUC | 0.676 | 0.740 |
| mean F1 @0.5 | 0.18 | 0.19 |
| test prevalence (preictal) | 0.5–11 %, mean 3.2 % | same |
| inner val AUC vs outer test AUC | 0.97–0.99 vs 0.46–0.90 | 0.83–0.98 vs 0.56–0.96 |

* `outer_shuffle: true` with `outer_method: LOO` only shuffles **interictal** windows
  before distributing them over folds (`data/splits.py` `leave_one_out`,
  `balanced_shuffled`). Each outer test fold still holds out one **whole preictal
  event**, so there is no preictal leakage.
* Models are trained 1:1 (undersampling) and tested at ~3 % prevalence; even a band-power
  logistic regression gets F1 ≈ 0.3 under these conditions (balanced F1 0.87–0.91).
* Event-level sensitivity is lenient (1 positive window in the ~180-window preictal
  period counts as detected), which is why sensitivity is ~1.0 while FPR/h is high.
* Non-shuffle sub-05 crashed (`inner_M=10` too large for a 34-window event) — fixed.

## 2. Bugs fixed (commit 7f12722)

| Bug | Fix |
|---|---|
| `robust_norm`/`instance_norm` absolute `eps=1e-5` ≈ signal scale in volts (std ≈ 2.4e-5) → under-normalised, quiet windows shrunk ≤0.4× | relative eps (`eps_mode="relative"` default; `"absolute"` reproduces old behaviour) |
| `amp: true` never read (training always fp32) | autocast + GradScaler (unscale before clip) in Trainer/TrainerMIL; **default `amp` changed True→False** so configs without the key keep fp32 |
| trailing batch of size 1 crashes BatchNorm | undersample loader drops a size-1 trailing *training* batch |
| `split_into_strata` raised on oversized `M` (sub-05) | clamp with warning |
| dead duplicate `training/dataloaders/undersample_loader.py` | removed |

Additions (opt-in / new columns only): `ce_stsenet` `dropout` kwarg (default 0);
`auprc` + `balanced_{f1,precision,recall,specificity}` columns in
`variant_summary.csv`; inner-validation-derived threshold
(`analysis/variant_summary_val_threshold.csv`, stored as `val_threshold` in
`raw_predictions.pkl`).

**Behaviour changes to be aware of:** the eps fix changes every run that uses
`robust_norm`/`instance_norm`; `amp: true` in a config now really uses fp16.

## 3. Label leak in moving-average post-processing (commit e874740)

`analysis/metrics.py::moving_average_segmented` splits smoothing regions on the
**true labels**. With chronological outer folds this changes results by ~0.001. With
shuffled outer folds the interictal test windows are scattered in time, so MA averages
random windows and FPR/h collapses artificially:

| run | MA | FPR/h label-segmented → time-segmented | MA-AUC |
|---|---|---|---|
| run002fix_B sub-11 (shuffled) | 10 | 0.00 → 3.29 | 0.992 → 0.972 |
| run002 shuffle sub-01 | 10 | 0.00 → 9.79 | 0.969 → 0.941 |
| wang2023 sub-11 (chronological) | 10 | 7.89 → 7.82 | 0.958 → 0.958 |

New flag `seizure-pred analyze --ma-segmentation time` segments by recording-time
contiguity (event_id + start time) and never reads labels; output goes to
`analysis_ma_time/`. Default unchanged. **All MA>1 numbers from shuffle-mode runs
(incl. full-dataset run002) are inflated; studies 001–007 (unshuffled) are essentially
unaffected.**

## 4. Experiments on sub-11 (2 preictal events → 2 outer folds)

Configs are copies of run002.yaml with `subject_id: "11"`; new data
`_fd_5s_prex5_intx3` = same preprocessing + `--interictal-oversample-factor 3`
(oversampled interictal windows are flagged `augmented` and excluded from metrics).

### 4a. Shuffle mode, one factor at a time

| run | change vs A | AUC | F1 | AUPRC | bal F1 | bal recall | FPR/h | fold AUCs |
|---|---|---|---|---|---|---|---|---|
| old code | — | 0.854 | 0.190 | 0.139 | 0.286 | 0.170 | 6.93 | .839 / .869 |
| A | fixed code | 0.851 | 0.175 | 0.132 | 0.257 | 0.150 | 6.19 | .827 / .875 |
| A repeat | none (determinism check) | 0.851 | 0.175 | 0.132 | 0.257 | 0.150 | 6.19 | identical |
| A_int | interictal ×3 | **0.907** | **0.275** | 0.231 | **0.424** | **0.278** | 8.00 | .889 / .926 |
| A_drop | dropout 0.3 | 0.872 | 0.189 | 0.138 | 0.283 | 0.167 | 6.93 | .843 / .900 |
| A_strata | inner_mode per_event_strata | 0.838 | 0.170 | 0.124 | 0.238 | 0.139 | 5.03 | .807 / .869 |
| B | all three | 0.901 | 0.270 | 0.243 | 0.331 | 0.200 | 3.29 | .878 / .925 |

Training is deterministic (A repeat is bit-identical). Interictal oversampling
explains B's gain; dropout helps a little; per-event inner strata slightly hurt.
Caveat: in shuffle mode, oversampled interictal training windows overlap original
interictal test windows, so part of that gain may be leakage — see 4c.

### 4b. Paper-style random split (leakage sanity check)

`outer_method: stratified`, `inner_method: stratified`, both shuffled (sklearn
StratifiedKFold over windows), otherwise as B. 100 % of test preictal windows have an
overlapping augmented neighbour in training.

| | AUC | F1 | AUPRC | bal F1 | bal recall | bal spec |
|---|---|---|---|---|---|---|
| B (held-out event) | 0.901 | 0.270 | 0.243 | 0.331 | 0.200 | 0.995 |
| C (random window split) | **0.9997** | **0.916** | 0.973 | 0.993 | 0.988 | 0.997 |

→ The pipeline reaches literature-level numbers as soon as preictal windows leak.

### 4c. Unshuffled (chronological) outer folds

Same as 4a but `outer_shuffle: false`, `inner_shuffle: false` (interictal test block is
chronologically separate, so oversampled interictal windows cannot overlap test windows
except at block edges). MA columns use the new label-free `--ma-segmentation time`.

| run | AUC | F1 | bal F1 | bal recall | bal spec | FPR/h | MA10 AUC | MA10 thr0.5 FPR/h (5-min refractory) | fold AUCs |
|---|---|---|---|---|---|---|---|---|---|
| A (noshuf) | 0.795 | 0.134 | 0.345 | 0.220 | 0.963 | 26.9 | 0.905 | 2.90 (0.39) | .773 / .817 |
| + interictal ×3 | **0.862** | 0.193 | **0.512** | **0.371** | 0.944 | 40.7 | **0.965** | 6.99 (0.93) | .830 / .895 |
| + interictal ×3 + dropout 0.3 | 0.861 | 0.200 | 0.475 | 0.326 | 0.962 | 27.2 | 0.960 | **2.71** (0.45) | .814 / .909 |

Event sensitivity is 1.0 for all three. Interictal oversampling improves the held-out-event
result without interictal shuffling too (AUC +0.07, balanced recall +0.15), so its gain is
not only leakage. A plausible mechanism: with ×3 overlapping interictal windows the
undersampling loader draws time-shifted negatives each epoch (a shift augmentation).
Dropout 0.3 on top keeps AUC and lowers false positives. Still one subject / two folds.

### 4d. Wang et al. 2023 protocol (MB-dMGC-CWTFFNet, TNSRE 31:4266)

256 Hz (`--no-downsample`, `si: 256`), 5 s clips, 15 min preictal, interictal ≥120 min
before onset (`--pre-buffer-minutes 105`), no oversampling, chronological LOO, MA 3
windows (15 s), threshold 0.6. Data: `_f_5s_pb23` (renamed after generation).
Deviation: the paper's 2 h post-ictal rule leaves chb11 with only one usable seizure
(seizures 2 and 3 fall within 2 h of seizure 1), so post-buffer was relaxed to 23 min
(3 folds); the paper's own Sn of 66.7/100 % for chb11 implies it also used 3 seizures.

| | AUC | Sn | FPR/h |
|---|---|---|---|
| Wang: their model, chb11 | 0.921 | 100 % | 0.037 |
| Wang: CE-stSENet baseline, chb11 | 0.897 | 66.7 % | 0.000 |
| Wang: CE-stSENet baseline, 18-patient mean | 0.857 | 86.0 % | 0.369 |
| ours, MA1 (raw window AUC) | 0.858 | 100 % | 31.6 |
| ours, MA3 thr 0.6 | 0.918 (smoothed) | 100 % | 7.96 (5-min refractory: 1.50) |

Our FPR/h counts every positive interictal window; the paper counts alarms, so
FPR/h is not comparable. Window AUC is in line with the paper's CE-stSENet baseline.

## 5. Literature check (`papers/`, `../../li2020.pdf`, `../../Dynamic_Multi_Graph...pdf`)

* High numbers come from window-level k-fold on overlapping windows with balanced test
  sets: MD-ResCapsNet (5-fold on 10 s windows / 5 s step; chb11 has one event, so all
  folds are within the same preictal period), SBTM (pooled % split / 10-fold).
  SeizureNet-KAN is internally contradictory.
* Honest leave-one-seizure-out papers land where we are: Wang 2023 (CE-stSENet baseline
  mean AUC 0.857), 3D-SERESNet (AUC 0.88–0.92).
* `li2020` (CE-stSENet) is a **seizure detection** paper, not prediction.

## 6. Open issues / recommendations

1. Report AUC, AUPRC, balanced metrics and *time-segmented* MA results; treat MA>1
   numbers from shuffle-mode runs as invalid.
2. Interictal oversampling is the single most effective change found — see 4c for
   whether it holds without interictal shuffling before adopting it for full runs.
   Memory: ×3 on sub-11 is ~3.3 GB in RAM; large subjects (e.g. sub-04, 36k interictal)
   would need ~5 GB+.
3. Preprocessing writes `event_stats.csv` without a suffix, and buffer settings are not
   encoded in the output suffix — two runs with different buffers overwrite each other.
4. `transforms/registry.py` still maps `instance_norm_fix_std` to a class that no longer
   exists (left for the owner).
5. Repo lacks: an alarm-event FPR/h (k-of-n + refractory as in 3D-SERESNet) and the
   "first seizure of a cluster only" rule.
6. `docs/studies.md` was not edited (it has uncommitted local changes).

## Artifacts

Runs: `runs/run002fix_A_sub11`, `run002fix_B_sub11`, `run002fix_C_sub11_randomsplit`,
`ablation_sub11_{A_repeat,intx3,dropout,strata}`, `ablation_sub11_noshuf_*`,
`wang2023_ce_stsenet_sub11`. New data (sub-11 only):
`processed_segments_fd_5s_prex5_intx3_float.npz`, `processed_segments_f_5s_pb{23,120}_float.npz`.
Incomplete run dir from a crashed attempt: `runs/ablation_sub11_intx3/<first stamp>` (only
`split_0`, can be deleted).
