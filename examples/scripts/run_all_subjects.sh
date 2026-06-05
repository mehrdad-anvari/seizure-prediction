#!/usr/bin/env bash
set -euo pipefail

# Train one or more models for multiple CHB-MIT subjects using the refactored CLI,
# and optionally run analysis for the latest split directory.

BASE_CONFIG="examples/config_prediction.yaml"
DATASET_DIR="data/BIDS_CHB-MIT"
SUFFIX="fd_5s_szx5_prex5"

# CHB-MIT subjects (12 often excluded due to channel inconsistencies)
SUBJECTS=(01 02 03 04 05 06 07 08 09 10 11 13 14 15 16 17 18 19 20 22 23 24)

# Model names must match the registry (`seizure-pred list models`)
MODELS=(
  "eegwavenet_tiny"
  # "ce_stsenet"
  # "mb_dmgc_cwtffnet"
)

# Training knobs
EPOCHS=30
BATCH_SIZE=64
LR=2e-3

for MODEL in "${MODELS[@]}"; do
  echo "================================================================"
  echo "Training model: $MODEL"
  echo "================================================================"

  for SUBJECT in "${SUBJECTS[@]}"; do
    echo "----------------------------------------------------------------"
    echo "Subject $SUBJECT"
    echo "----------------------------------------------------------------"

    OVERRIDE_FILE=$(mktemp -t seizure_pred_override.XXXXXX.yaml)
    cat > "$OVERRIDE_FILE" <<EOF
run_name: "${MODEL}_sub${SUBJECT}_$SUFFIX"
save_dir: runs
epochs: ${EPOCHS}

data:
  dataset_dir: "${DATASET_DIR}"
  subject_id: "${SUBJECT}"
  batch_size: ${BATCH_SIZE}
  suffix: "${SUFFIX}"

model:
  name: "${MODEL}"

optim:
  lr: ${LR}
EOF

    # Train
    if ! python -m seizure_pred.cli.main train --config "$BASE_CONFIG" --override "$OVERRIDE_FILE"; then
      echo "ERROR: training failed for model=$MODEL subject=$SUBJECT"
      rm -f "$OVERRIDE_FILE"
      continue
    fi
    rm -f "$OVERRIDE_FILE"

    # Find latest split dir under runs/<run_name>/<stamp>/split_0
    LATEST_SPLIT_DIR=$(ls -td runs/${MODEL}_sub${SUBJECT}_${SUFFIX}/*/split_0 2>/dev/null | head -n 1 || true)
    if [ -d "$LATEST_SPLIT_DIR" ]; then
      echo "--> Analyzing: $LATEST_SPLIT_DIR"
      python examples/scripts/analyze_results.py --run-dir "$LATEST_SPLIT_DIR" || true
    else
      echo "WARNING: could not locate split dir for analysis."
    fi
  done
done

echo "Done."
