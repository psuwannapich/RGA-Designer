#!/bin/bash
#SBATCH --job-name=arg_train_rm
#SBATCH --output=logs/train_rm_%j.out
#SBATCH --error=logs/train_rm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH -C volta32
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# RLHF Phase 2 — train the GNN reward model on collected preference pairs.
#
# Usage:
#   sbatch slurm/train_rm.sh
#
# This phase does NOT need an LLM — GPU is used only for GNN training.
# Override defaults with env vars before submitting:
#   PREFERENCE_DIR=rlhf_data/gsm8k
#   RM_CHECKPOINT=rlhf_checkpoints/gsm8k/reward_model.pth
# ---------------------------------------------------------------------------

set -euo pipefail

PREFERENCE_DIR="${PREFERENCE_DIR:-rlhf_data/gsm8k}"
RM_CHECKPOINT="${RM_CHECKPOINT:-rlhf_checkpoints/gsm8k/reward_model.pth}"
RM_EPOCHS="${RM_EPOCHS:-20}"
RM_LR="${RM_LR:-1e-4}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-32}"
RM_HIDDEN_DIM="${RM_HIDDEN_DIM:-256}"
RM_OUTPUT_DIM="${RM_OUTPUT_DIM:-128}"
RM_VAL_FRACTION="${RM_VAL_FRACTION:-0.1}"

mkdir -p logs

echo "========================================"
echo "Job ID          : $SLURM_JOB_ID"
echo "Node            : $SLURM_NODELIST"
echo "Preference dir  : $PREFERENCE_DIR"
echo "RM checkpoint   : $RM_CHECKPOINT"
echo "Epochs          : $RM_EPOCHS"
echo "Started at      : $(date)"
echo "========================================"

uv run rlhf \
    --phase            train_rm \
    --preference_dir   "$PREFERENCE_DIR" \
    --rm_checkpoint    "$RM_CHECKPOINT" \
    --rm_epochs        "$RM_EPOCHS" \
    --rm_lr            "$RM_LR" \
    --rm_batch_size    "$RM_BATCH_SIZE" \
    --rm_hidden_dim    "$RM_HIDDEN_DIM" \
    --rm_output_dim    "$RM_OUTPUT_DIM" \
    --rm_val_fraction  "$RM_VAL_FRACTION"

echo "Finished at: $(date)"
