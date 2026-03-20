#!/bin/bash
#SBATCH --job-name=arg_cold_start
#SBATCH --output=logs/cold_start_%j.out
#SBATCH --error=logs/cold_start_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# Adjust partition to match your HPC cluster (e.g. --partition=gpu)
# #SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# Cold-start dataset generation using a HuggingFace model directly.
#
# Usage:
#   sbatch slurm/cold_start.sh
#
# Override defaults with env vars before submitting:
#   HF_MODEL=google/gemma-3-4b-it
#   DATASET=gsm8k
#   NUM_TASKS=40
#   OUTPUT_DIR=ColdStartData_gemma_gsm8k
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- User-configurable defaults -------------------------------------------
HF_MODEL="${HF_MODEL:-google/gemma-3-4b-it}"
DATASET="${DATASET:-gsm8k}"
DATASET_JSON="${DATASET_JSON:-datasets/${DATASET}/${DATASET}.jsonl}"
NUM_TASKS="${NUM_TASKS:-40}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_ROUNDS="${NUM_ROUNDS:-1}"
MIN_AGENTS="${MIN_AGENTS:-3}"
MAX_AGENTS="${MAX_AGENTS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-ColdStartData_hf_${DATASET}}"
SEED="${SEED:-42}"

# Optional: point to a shared model cache on the cluster's scratch filesystem
# export HF_MODEL_CACHE=/scratch/$USER/hf_cache
export HF_MODEL_CACHE="${HF_MODEL_CACHE:-$HOME/.cache/huggingface}"

# ---------------------------------------------------------------------------
mkdir -p logs

echo "========================================"
echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURM_NODELIST"
echo "HF Model      : $HF_MODEL"
echo "Dataset       : $DATASET  ($NUM_TASKS tasks)"
echo "Output dir    : $OUTPUT_DIR"
echo "Started at    : $(date)"
echo "========================================"

uv run cold-start \
    --dataset        "$DATASET" \
    --dataset_json   "$DATASET_JSON" \
    --llm_name       "$HF_MODEL" \
    --output_dir     "$OUTPUT_DIR" \
    --num_tasks      "$NUM_TASKS" \
    --batch_size     "$BATCH_SIZE" \
    --num_rounds     "$NUM_ROUNDS" \
    --min_agents     "$MIN_AGENTS" \
    --max_agents     "$MAX_AGENTS" \
    --seed           "$SEED"

echo "Finished at: $(date)"
