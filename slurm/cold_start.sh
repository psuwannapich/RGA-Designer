#!/bin/bash
#SBATCH --job-name=arg_cold_start
#SBATCH -p gpu
#SBATCH --output=logs/cold_start_%j.out
#SBATCH --error=logs/cold_start_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH -C volta32
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# Adjust partition to match your HPC cluster (e.g. --partition=gpu)
# #SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# Cold-start dataset generation using a HuggingFace model directly.
#
# Usage:
#   sbatch slurm/cold_start.sh
#
# Override defaults with env vars before submitting:
#   HF_MODEL=Qwen/Qwen3-8B
#   DATASET=gsm8k
#   NUM_TASKS=0
#   OUTPUT_DIR=ColdStartData_gemma_gsm8k
# ---------------------------------------------------------------------------

set -euo pipefail

# Load CUDA modules so libcudnn.so is on LD_LIBRARY_PATH before torch imports.
_SLURM_DIR="${SLURM_SUBMIT_DIR:+${SLURM_SUBMIT_DIR}/slurm}"
source "${_SLURM_DIR:-$(dirname "${BASH_SOURCE[0]}")}/setup_cuda.sh"

# ---- User-configurable defaults -------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
MODEL_SLUG="${HF_MODEL//\//-}"                     # Qwen/Qwen3-8B → Qwen-Qwen3-8B
DATASET="${DATASET:-gsm8k}"
DATASET_JSON="${DATASET_JSON:-benchmark_datasets/${DATASET}/${DATASET}.jsonl}"
NUM_TASKS="${NUM_TASKS:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_ROUNDS="${NUM_ROUNDS:-1}"
MIN_AGENTS="${MIN_AGENTS:-3}"
MAX_AGENTS="${MAX_AGENTS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_SLUG}/ColdStartData/${DATASET}}"
SEED="${SEED:-42}"


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
