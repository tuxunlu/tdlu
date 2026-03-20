#!/bin/bash
# Evaluate STAMP model checkpoints across folds.
# Auto-discovers best-*.ckpt in runs/*foldN*/version_0/checkpoints/

cd "$(dirname "$0")"

CONFIG="config/config_torchio_3bins_tdlu_stamp_one_view_10folds_zero_extreme_meta_avg_cross_attention.yaml"
RUN_BASE="/beacon-projects/mammography/tdlu/runs"

# Find best checkpoint for fold N (path must contain "foldN/" to avoid fold0 matching fold01)
ckpt() {
    find "$RUN_BASE" -path "*fold${1}/*" -path "*checkpoints*" -name "best-*.ckpt" 2>/dev/null | head -1
}

CKPT_0=$(ckpt 0)
CKPT_1=$(ckpt 1)
CKPT_2=$(ckpt 2)
CKPT_3=$(ckpt 3)
CKPT_4=$(ckpt 4)

# Debug: show found checkpoints
echo "Found checkpoints:"
for i in 0 1 2 3 4; do
    c=$(ckpt $i)
    echo "  fold $i: ${c:-<none>}"
done

python test_all_folds_stamp.py \
    --config "$CONFIG" \
    --checkpoint_0 "$CKPT_0" \
    --checkpoint_1 "$CKPT_1" \
    --checkpoint_2 "$CKPT_2" \
    --checkpoint_3 "$CKPT_3" \
    --checkpoint_4 "$CKPT_4"
