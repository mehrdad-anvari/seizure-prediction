#!/usr/bin/env bash
set -euo pipefail

# Run a small grid of training runs by varying the processed-data suffix.
#
# This script uses the new CLI:
#   python -m seizure_pred.cli.main train --config <base> --override <override>
#
# Base config (edit if you want a different default):
BASE_CONFIG="examples/config_prediction.yaml"

# Dataset/model settings
DATASET_DIR="data/BIDS_CHB-MIT"
SUBJECT_ID="01"
MODEL="ce_stsenet"     # Use `python -m seizure_pred.cli.main list models` to see names

# Training knobs
EPOCHS=20
BATCH_SIZE=64
LR=1e-3

# In the old repo this script varied preprocessing flags (normalization/ICA/filter).
# In the refactored repo, preprocessing is a separate step and its outputs are identified by `suffix`.
# So we vary the suffix here.
SUFFIXES=(
  "fd_5s_szx5_prex5"
  # add your own processed variants here
)

mkdir -p "runs"

for SUFFIX in "${SUFFIXES[@]}"; do
  echo "================================================"
  echo "Training: subject=${SUBJECT_ID} model=${MODEL} suffix=${SUFFIX}"
  echo "================================================"

  # Build a temporary override file.
  OVERRIDE_FILE=$(mktemp -t seizure_pred_override.XXXXXX.yaml)
  cat > "$OVERRIDE_FILE" <<EOF
run_name: "exp_${MODEL}_sub${SUBJECT_ID}_${SUFFIX}"
save_dir: runs
epochs: ${EPOCHS}

data:
  dataset_dir: "${DATASET_DIR}"
  subject_id: "${SUBJECT_ID}"
  batch_size: ${BATCH_SIZE}
  suffix: "${SUFFIX}"

model:
  name: "${MODEL}"

optim:
  lr: ${LR}
EOF

  python -m seizure_pred.cli.main train --config "$BASE_CONFIG" --override "$OVERRIDE_FILE"

  rm -f "$OVERRIDE_FILE"
  echo ""
done

echo "================================================"
echo "All experiments completed."
echo "================================================"
