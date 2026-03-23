#!/bin/bash
#SBATCH --job-name=arg_train_rm
#SBATCH --output=logs/train_rm_%A_%a.out
#SBATCH --error=logs/train_rm_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5          # 0=gsm8k 1=aqua 2=multiarith 3=svamp 4=humaneval 5=mmlu
#SBATCH -p gpu
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# RLHF Phase 2 — train GNN reward model on collected preference pairs.
# No LLM needed — 1 GPU is sufficient.
#
# Optional env vars:
#   PREFERENCE_ROOT   root dir for preference data  (default: rlhf_data)
#   RM_ROOT           root dir for RM checkpoints   (default: rlhf_checkpoints)
#   RM_EPOCHS         training epochs               (default: 20)
#
# Usage:
#   sbatch slurm/train_rm.sh
#   sbatch --array=0 slurm/train_rm.sh   # gsm8k only
# ---------------------------------------------------------------------------

set -euo pipefail

DATASETS=(gsm8k aqua multiarith svamp humaneval mmlu)

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT_ROOT/logs"

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

PREFERENCE_ROOT="${PREFERENCE_ROOT:-rlhf_data}"
RM_ROOT="${RM_ROOT:-rlhf_checkpoints}"
PREFERENCE_DIR="$PROJECT_ROOT/${PREFERENCE_ROOT}/${DATASET}"
RM_CHECKPOINT="$PROJECT_ROOT/${RM_ROOT}/${DATASET}/reward_model.pth"
RM_EPOCHS="${RM_EPOCHS:-20}"
RM_LR="${RM_LR:-1e-4}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-32}"
RM_HIDDEN_DIM="${RM_HIDDEN_DIM:-256}"
RM_OUTPUT_DIM="${RM_OUTPUT_DIM:-128}"
RM_VAL_FRACTION="${RM_VAL_FRACTION:-0.1}"

echo "========================================"
echo "Job ID          : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node            : $SLURM_NODELIST"
echo "Dataset         : $DATASET"
echo "Preference dir  : $PREFERENCE_DIR"
echo "RM checkpoint   : $RM_CHECKPOINT"
echo "Epochs          : $RM_EPOCHS"
echo "Started at      : $(date)"
echo "========================================"

cd "$PROJECT_ROOT"

uv run rlhf \
    --dataset          "$DATASET" \
    --phase            train_rm \
    --preference_dir   "$PREFERENCE_DIR" \
    --rm_checkpoint    "$RM_CHECKPOINT" \
    --rm_epochs        "$RM_EPOCHS" \
    --rm_lr            "$RM_LR" \
    --rm_batch_size    "$RM_BATCH_SIZE" \
    --rm_hidden_dim    "$RM_HIDDEN_DIM" \
    --rm_output_dim    "$RM_OUTPUT_DIM" \
    --rm_val_fraction  "$RM_VAL_FRACTION"

echo "========================================"
echo "train_rm complete for $DATASET"
echo "Finished at: $(date)"
echo "========================================"
