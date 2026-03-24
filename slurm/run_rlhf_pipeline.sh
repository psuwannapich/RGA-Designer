#!/bin/bash
# ---------------------------------------------------------------------------
# Full RLHF pipeline: collect → train_rm → train_policy
#
# Submits three dependent job arrays.  Each stage waits for all jobs in the
# previous stage to succeed before starting.
#
# Usage:
#   bash slurm/run_rlhf_pipeline.sh
#
# Optional env vars:
#   HF_MODEL          HuggingFace model ID          (default: Qwen/Qwen3-8B)
#   NUM_TASKS         tasks per dataset for collect  (default: 100)
#   DATASETS_ARRAY    Slurm array spec               (default: 0-5 = all)
#   NUM_GPUS          GPUs per LLM job               (default: 2)
#   CHECKPOINT_ROOT   Phase-1/2 model checkpoint dir (default: checkpoints)
#   PREFERENCE_ROOT   preference data root           (default: rlhf_data)
#   RM_ROOT           reward model checkpoint root   (default: rlhf_checkpoints)
#   POLICY_ROOT       policy checkpoint root         (default: rlhf_checkpoints)
#
# Example — run only gsm8k (index 0):
#   DATASETS_ARRAY=0 bash slurm/run_rlhf_pipeline.sh
# ---------------------------------------------------------------------------

set -euo pipefail

HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
NUM_TASKS="${NUM_TASKS:-100}"
DATASETS_ARRAY="${DATASETS_ARRAY:-0-5}"
NUM_GPUS="${NUM_GPUS:-2}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
PREFERENCE_ROOT="${PREFERENCE_ROOT:-rlhf_data}"
RM_ROOT="${RM_ROOT:-rlhf_checkpoints}"
POLICY_ROOT="${POLICY_ROOT:-rlhf_checkpoints}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.."; pwd)"
mkdir -p "$PROJECT_ROOT/logs"

echo "================================================"
echo "ARG-Designer RLHF Pipeline"
echo "Model         : $HF_MODEL"
echo "Datasets array: $DATASETS_ARRAY"
echo "Num tasks     : $NUM_TASKS"
echo "GPUs per LLM job: $NUM_GPUS  (train_rm always uses 1)"
echo "================================================"

# ---- Stage 1: Collect preference data --------------------------------------
echo ""
echo "[Stage 1] Submitting collect jobs (array: $DATASETS_ARRAY) ..."
COLLECT_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --gres="gpu:${NUM_GPUS}" \
    --export=ALL,HF_MODEL="$HF_MODEL",NUM_TASKS="$NUM_TASKS",PREFERENCE_ROOT="$PREFERENCE_ROOT" \
    "$PROJECT_ROOT/slurm/collect.sh" \
    | awk '{print $NF}')
echo "  Collect job ID: $COLLECT_JOB"

# ---- Stage 2: Train reward model -------------------------------------------
echo ""
echo "[Stage 2] Submitting train_rm jobs (depends on $COLLECT_JOB) ..."
TRAIN_RM_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --dependency=afterok:"$COLLECT_JOB" \
    --gres="gpu:1" \
    --export=ALL,HF_MODEL="$HF_MODEL",PREFERENCE_ROOT="$PREFERENCE_ROOT",RM_ROOT="$RM_ROOT" \
    "$PROJECT_ROOT/slurm/train_rm.sh" \
    | awk '{print $NF}')
echo "  train_rm job ID: $TRAIN_RM_JOB"

# ---- Stage 3: Train policy -------------------------------------------------
echo ""
echo "[Stage 3] Submitting train_policy jobs (depends on $TRAIN_RM_JOB) ..."
POLICY_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --dependency=afterok:"$TRAIN_RM_JOB" \
    --gres="gpu:${NUM_GPUS}" \
    --export=ALL,HF_MODEL="$HF_MODEL",CHECKPOINT_ROOT="$CHECKPOINT_ROOT",RM_ROOT="$RM_ROOT",POLICY_ROOT="$POLICY_ROOT",NUM_TASKS="$NUM_TASKS" \
    "$PROJECT_ROOT/slurm/train_policy.sh" \
    | awk '{print $NF}')
echo "  train_policy job ID: $POLICY_JOB"

echo ""
echo "================================================"
echo "All RLHF stages submitted successfully."
echo ""
echo "  Stage 1  collect      : job $COLLECT_JOB"
echo "  Stage 2  train_rm     : job $TRAIN_RM_JOB   (waits for $COLLECT_JOB)"
echo "  Stage 3  train_policy : job $POLICY_JOB     (waits for $TRAIN_RM_JOB)"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f $PROJECT_ROOT/logs/collect_${COLLECT_JOB}_*.out"
echo "  tail -f $PROJECT_ROOT/logs/train_rm_${TRAIN_RM_JOB}_*.out"
echo "  tail -f $PROJECT_ROOT/logs/train_policy_${POLICY_JOB}_*.out"
echo "================================================"
