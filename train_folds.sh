#!/bin/bash
# Run cross-validation folds sequentially on GPU 0 (explicit commands; no aliases, no loops).
#
# Usage:
#   ./train_folds.sh CONFIG_PATH
#
# Example:
#   ./train_folds.sh config/config_torchio_3bins_tdlu_stamp_one_view_10folds_zero_extreme_meta_avg_cross_attention.yaml

set -euo pipefail

CONFIG_PATH="${1:?Usage: $0 CONFIG_PATH}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$CONFIG_PATH" ]]; then
  if [[ -f "$SCRIPT_DIR/$CONFIG_PATH" ]]; then
    CONFIG_PATH="$SCRIPT_DIR/$CONFIG_PATH"
  else
    echo "Error: Config not found: $CONFIG_PATH"
    exit 1
  fi
fi

mkdir -p logs

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_DIR/main.py" --config_path "$CONFIG_PATH" --cross_val_fold 0 > logs/fold0_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_DIR/main.py" --config_path "$CONFIG_PATH" --cross_val_fold 1 > logs/fold1_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python "$SCRIPT_DIR/main.py" --config_path "$CONFIG_PATH" --cross_val_fold 2 > logs/fold2_gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python "$SCRIPT_DIR/main.py" --config_path "$CONFIG_PATH" --cross_val_fold 3 > logs/fold3_gpu0.log 2>&1

