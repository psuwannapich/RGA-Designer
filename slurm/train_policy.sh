#!/bin/bash
#SBATCH --job-name=arg_train_policy
#SBATCH --output=logs/train_policy_%A_%a.out
#SBATCH --error=logs/train_policy_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --array=0-5          # 0=gsm8k 1=aqua 2=multiarith 3=svamp 4=humaneval 5=mmlu
#SBATCH -p gpu
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# RLHF Phase 3 — fine-tune ARGDesigner (policy) via REINFORCE + KL penalty.
#
# Optional env vars:
#   HF_MODEL          HuggingFace model ID        (default: Qwen/Qwen3-8B)
#   CHECKPOINT_ROOT   Phase-1/2 checkpoint root   (default: checkpoints)
#   RM_ROOT           reward model root           (default: rlhf_checkpoints)
#   POLICY_ROOT       policy output root          (default: rlhf_checkpoints)
#   NUM_TASKS         tasks to sample             (default: 100)
#
# Usage:
#   sbatch slurm/train_policy.sh
#   sbatch --array=0 slurm/train_policy.sh   # gsm8k only
# ---------------------------------------------------------------------------

set -euo pipefail

DATASETS=(gsm8k aqua multiarith svamp humaneval mmlu)
DATASET_JSONS=(
    "datasets/gsm8k/gsm8k.jsonl"
    "datasets/AQuA/AQuA.jsonl"
    "datasets/MultiArith/MultiArith.json"
    "datasets/SVAMP/SVAMP.json"
    "datasets/humaneval/humaneval-py.jsonl"
    "datasets/MMLU/data"
)

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT_ROOT/logs"

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$SLURM_ARRAY_TASK_ID]}"

HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
RM_ROOT="${RM_ROOT:-rlhf_checkpoints}"
POLICY_ROOT="${POLICY_ROOT:-rlhf_checkpoints}"
MODEL_DIR="$PROJECT_ROOT/${CHECKPOINT_ROOT}/${DATASET}"
RM_CHECKPOINT="$PROJECT_ROOT/${RM_ROOT}/${DATASET}/reward_model.pth"
POLICY_CHECKPOINT="$PROJECT_ROOT/${POLICY_ROOT}/${DATASET}/policy_rlhf.pth"
NUM_TASKS="${NUM_TASKS:-100}"
POLICY_EPOCHS="${POLICY_EPOCHS:-10}"
POLICY_LR="${POLICY_LR:-1e-5}"
KL_COEFF="${KL_COEFF:-0.1}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-2}"
SEED="${SEED:-42}"

echo "========================================"
echo "Job ID           : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node             : $SLURM_NODELIST"
echo "Dataset          : $DATASET"
echo "Model dir        : $MODEL_DIR"
echo "RM checkpoint    : $RM_CHECKPOINT"
echo "Policy checkpoint: $POLICY_CHECKPOINT"
echo "Num tasks        : $NUM_TASKS"
echo "Policy epochs    : $POLICY_EPOCHS"
echo "Started at       : $(date)"
echo "========================================"

cd "$PROJECT_ROOT"

uv run rlhf \
    --dataset            "$DATASET" \
    --phase              train_policy \
    --llm_name           "$HF_MODEL" \
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

echo "========================================"
echo "train_policy complete for $DATASET"
echo "Finished at: $(date)"
echo "========================================"
