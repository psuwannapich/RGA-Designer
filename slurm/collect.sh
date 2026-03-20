#!/bin/bash
#SBATCH --job-name=arg_collect
#SBATCH --output=logs/collect_%j.out
#SBATCH --error=logs/collect_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
# #SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# RLHF Phase 1 — collect preference pairs using a HuggingFace model directly.
#
# Usage:
#   sbatch slurm/collect.sh
#
# Override defaults with env vars before submitting:
#   HF_MODEL=google/gemma-3-4b-it
#   NUM_TASKS=100
#   PREFERENCE_DIR=rlhf_data/gsm8k
# ---------------------------------------------------------------------------

set -euo pipefail

HF_MODEL="${HF_MODEL:-google/gemma-3-4b-it}"
DATASET_JSON="${DATASET_JSON:-datasets/gsm8k/gsm8k.jsonl}"
NUM_TASKS="${NUM_TASKS:-100}"
PREFERENCE_DIR="${PREFERENCE_DIR:-rlhf_data/gsm8k}"
MIN_AGENTS="${MIN_AGENTS:-2}"
MAX_AGENTS="${MAX_AGENTS:-4}"
W_CORRECT="${W_CORRECT:-0.6}"
W_SIZE="${W_SIZE:-0.2}"
W_TOKEN="${W_TOKEN:-0.2}"
PAIR_MARGIN="${PAIR_MARGIN:-0.05}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-20}"
SEED="${SEED:-42}"

export HF_MODEL_CACHE="${HF_MODEL_CACHE:-$HOME/.cache/huggingface}"

mkdir -p logs

echo "========================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURM_NODELIST"
echo "HF Model      : $HF_MODEL"
echo "Num tasks     : $NUM_TASKS"
echo "Preference dir: $PREFERENCE_DIR"
echo "Started at    : $(date)"
echo "========================================"

uv run rlhf \
    --phase          collect \
    --llm_name       "$HF_MODEL" \
    --dataset_json   "$DATASET_JSON" \
    --num_tasks      "$NUM_TASKS" \
    --preference_dir "$PREFERENCE_DIR" \
    --min_agents     "$MIN_AGENTS" \
    --max_agents     "$MAX_AGENTS" \
    --w_correct      "$W_CORRECT" \
    --w_size         "$W_SIZE" \
    --w_token        "$W_TOKEN" \
    --pair_margin    "$PAIR_MARGIN" \
    --checkpoint_every "$CHECKPOINT_EVERY" \
    --seed           "$SEED"

echo "Finished at: $(date)"
