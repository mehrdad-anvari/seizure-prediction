#!/usr/bin/env bash
set -euo pipefail

# Example end-to-end training runs using the new library CLI.
#
# Prerequisite (repo root):
#   pip install -e ".[cli]"
#
# This script demonstrates:
#   - generating a base config
#   - training with overrides

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_DIR="examples/scripts/_configs"
mkdir -p "$CONFIG_DIR"

BASE_CFG="$CONFIG_DIR/base.yaml"

# 1) Create a starter config
python examples/scripts/config_template.py \
  --out "$BASE_CFG" \
  --task prediction \
  --dataset-dir "data/BIDS_CHB-MIT" \
  --subject "01" \
  --model "ce_stsenet"

echo "[run_example] Base config written to: $BASE_CFG"

# 2) Quick run (short epochs)
cat > "$CONFIG_DIR/override_quick.yaml" <<'YAML'
epochs: 20
data:
  batch_size: 64
model:
  name: ce_stsenet
optim:
  lr: 1.0e-3
YAML

python -m seizure_pred.cli.main train --config "$BASE_CFG" --override "$CONFIG_DIR/override_quick.yaml" --split-index 0 --n-folds 5

# 3) Another run with a different model + subject
cat > "$CONFIG_DIR/override_eegnet.yaml" <<'YAML'
epochs: 50
data:
  subject_id: "02"
  batch_size: 32
model:
  name: eegnet
optim:
  lr: 5.0e-4
YAML

python -m seizure_pred.cli.main train --config "$BASE_CFG" --override "$CONFIG_DIR/override_eegnet.yaml" --split-index 0 --n-folds 5

echo "[run_example] Done. See runs/ for outputs."
