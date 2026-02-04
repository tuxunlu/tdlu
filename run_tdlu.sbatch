#!/bin/bash
#SBATCH --job-name=tdlu_train
#SBATCH --partition=beacon
#SBATCH --qos=high
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2        # 1 task per GPU (DDP, 2 ranks total)
#SBATCH --gres=gpu:2               # 2 GPUs on the node
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4          # tune for your DataLoader workers
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --export=ALL

# Fail fast and show commands (optional)
set -euo pipefail

echo "[$(date)] Job $SLURM_JOB_ID starting on $SLURM_NODELIST"
echo "SLURM_NTASKS=$SLURM_NTASKS  GPUs requested=$(echo $CUDA_VISIBLE_DEVICES 2>/dev/null || true)"

# If your code lives somewhere specific, uncomment and set:
cd /beacon-scratch/tuxunlu/git/tdlu

# Activate your environment
source activate.sh

# export PYTHONPATH="/path/to/my/custom/module:$PYTHONPATH"

# Launch your training
# Use srun under sbatch so SLURM wiring (env, rank, etc.) is passed through.
srun python main.py \
  --config_path config_torchio_2bins_tdlu_four_views_stacked_10folds_zero_extreme_meta_avg_cross_attention.yaml \
  --cross_val_fold 0 \

echo "[$(date)] Job $SLURM_JOB_ID finished"
