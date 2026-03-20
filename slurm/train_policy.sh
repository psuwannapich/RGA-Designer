#!/bin/bash
#SBATCH --job-name=arg_train_policy
#SBATCH --output=logs/train_policy_%j.out
#SBATCH --error=logs/train_policy_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
# #SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# RLHF Phase 3 — fine-tune ARGDesigner (policy) via REINFORCE + KL penalty.
#
# Usage:
#   sbatch slurm/train_policy.sh
#
# Requires a pretrained ARGDesigner checkpoint (--model_dir) and a trained
# reward model checkpoint (--rm_checkpoint) from Phase 2.
#
# Override defaults with env vars before submitting:
#   MODEL_DIR=ColdStartData_hf_gsm8k
#   RM_CHECKPOINT=rlhf_checkpoints/gsm8k/reward_model.pth
#   POLICY_CHECKPOINT=rlhf_checkpoints/gsm8k/policy_rlhf.pth
# ---------------------------------------------------------------------------

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-}"          # REQUIRED — path to pretrained ARGDesigner
RM_CHECKPOINT="${RM_CHECKPOINT:-rlhf_checkpoints/gsm8k/reward_model.pth}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:-rlhf_checkpoints/gsm8k/policy_rlhf.pth}"
DATASET_JSON="${DATASET_JSON:-datasets/gsm8k/gsm8k.jsonl}"
NUM_TASKS="${NUM_TASKS:-100}"
POLICY_EPOCHS="${POLICY_EPOCHS:-10}"
POLICY_LR="${POLICY_LR:-1e-5}"
KL_COEFF="${KL_COEFF:-0.1}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-2}"
SEED="${SEED:-42}"

if [[ -z "$MODEL_DIR" ]]; then
    echo "ERROR: MODEL_DIR must be set to the pretrained ARGDesigner directory."
    echo "  e.g.: MODEL_DIR=ColdStartData_hf_gsm8k sbatch slurm/train_policy.sh"
    exit 1
fi

mkdir -p logs

echo "========================================"
echo "Job ID           : $SLURM_JOB_ID"
echo "Node             : $SLURM_NODELIST"
echo "Model dir        : $MODEL_DIR"
echo "RM checkpoint    : $RM_CHECKPOINT"
echo "Policy checkpoint: $POLICY_CHECKPOINT"
echo "Num tasks        : $NUM_TASKS"
echo "Policy epochs    : $POLICY_EPOCHS"
echo "Started at       : $(date)"
echo "========================================"

uv run rlhf \
    --phase              train_policy \
    --model_dir          "$MODEL_DIR" \
    --rm_checkpoint      "$RM_CHECKPOINT" \
    --policy_checkpoint  "$POLICY_CHECKPOINT" \
    --dataset_json       "$DATASET_JSON" \
    --num_tasks          "$NUM_TASKS" \
    --policy_epochs      "$POLICY_EPOCHS" \
    --policy_lr          "$POLICY_LR" \
    --kl_coeff           "$KL_COEFF" \
    --samples_per_task   "$SAMPLES_PER_TASK" \
    --seed               "$SEED"

echo "Finished at: $(date)"
