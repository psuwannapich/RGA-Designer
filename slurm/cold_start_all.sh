#!/bin/bash
#SBATCH --job-name=arg_cold_start_all
#SBATCH --output=logs/cold_start_%A_%a.out
#SBATCH --error=logs/cold_start_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --array=0-5          # one task per dataset (see table below)
#SBATCH -p gpu
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# Cold-start dataset generation for ALL supported datasets.
#
# Array index → dataset mapping:
#   0  gsm8k      benchmark_datasets/gsm8k/gsm8k.jsonl
#   1  aqua       benchmark_datasets/AQuA/AQuA.jsonl
#   2  multiarith benchmark_datasets/MultiArith/MultiArith.json
#   3  svamp      benchmark_datasets/SVAMP/SVAMP.json
#   4  humaneval  benchmark_datasets/humaneval/humaneval-py.jsonl
#   5  mmlu       benchmark_datasets/MMLU/data  (CSV dir; auto-downloaded if missing)
#
# Usage — submit all six jobs in parallel:
#   sbatch slurm/cold_start_all.sh
#
# Submit a subset (e.g. only math datasets):
#   sbatch --array=0-3 slurm/cold_start_all.sh
#
# Submit a single dataset by index (e.g. mmlu only):
#   sbatch --array=5 slurm/cold_start_all.sh
#
# Override global defaults before submitting:
#   HF_MODEL=Qwen/Qwen3-8B NUM_TASKS=0 sbatch slurm/cold_start_all.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- Dataset registry (indices must match --array range above) ------------
DATASETS=(
    gsm8k        # 0
    aqua         # 1
    multiarith   # 2
    svamp        # 3
    humaneval    # 4
    mmlu         # 5
)
JSONLS=(
    "benchmark_datasets/gsm8k/gsm8k.jsonl"            # 0
    "benchmark_datasets/AQuA/AQuA.jsonl"              # 1
    "benchmark_datasets/MultiArith/MultiArith.json"   # 2
    "benchmark_datasets/SVAMP/SVAMP.json"             # 3
    "benchmark_datasets/humaneval/humaneval-py.jsonl" # 4
    "benchmark_datasets/MMLU/data"                    # 5 — directory, not a single file
)

# Per-dataset agent counts (from original cold-start scripts)
#   gsm8k/aqua/multiarith/svamp: max 4  (range 3-4)
#   humaneval:                   max 5  (range 3-5)
#   mmlu:                        max 6  (range 3-6)
DATASET_MIN_AGENTS=(3 3 3 3 3 3)   # index: 0=gsm8k 1=aqua 2=multiarith 3=svamp 4=humaneval 5=mmlu
DATASET_MAX_AGENTS=(4 4 4 4 5 6)   # index: 0=gsm8k 1=aqua 2=multiarith 3=svamp 4=humaneval 5=mmlu

# ---- Global defaults (override via env vars) ------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
MODEL_SLUG="${HF_MODEL//\//-}"                     # Qwen/Qwen3-8B → Qwen-Qwen3-8B
NUM_TASKS="${NUM_TASKS:-0}"
NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_ROUNDS="${NUM_ROUNDS:-1}"
SEED="${SEED:-42}"

# ---- Select this task's dataset -------------------------------------------
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATASET_JSON="${JSONLS[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="${MODEL_SLUG}/ColdStartData/${DATASET}"

# Per-dataset agent range (env vars override if set)
MIN_AGENTS="${MIN_AGENTS:-${DATASET_MIN_AGENTS[$SLURM_ARRAY_TASK_ID]}}"
MAX_AGENTS="${MAX_AGENTS:-${DATASET_MAX_AGENTS[$SLURM_ARRAY_TASK_ID]}}"

mkdir -p logs

echo "========================================"
echo "Job ID        : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node          : $SLURM_NODELIST"
echo "HF Model      : $HF_MODEL"
echo "Dataset       : $DATASET  (num_tasks=${NUM_TASKS}, 0=all base tasks)"
echo "Dataset JSON  : $DATASET_JSON"
echo "Output dir    : $OUTPUT_DIR"
echo "Agents        : min=$MIN_AGENTS  max=$MAX_AGENTS"
echo "Started at    : $(date)"
echo "========================================"

# Download MMLU data if this is the mmlu job and the data dir is missing
if [[ "$DATASET" == "mmlu" && ! -d "$DATASET_JSON/test" ]]; then
    echo "MMLU data not found — running download script..."
    uv run python benchmark_datasets/MMLU/download.py
    echo "MMLU download complete."
fi

uv run cold-start \
    --dataset        "$DATASET" \
    --dataset_json   "$DATASET_JSON" \
    --llm_name       "$HF_MODEL" \
    --output_dir     "$OUTPUT_DIR" \
    --num_tasks      "$NUM_TASKS" \
    --num_iterations "$NUM_ITERATIONS" \
    --batch_size     "$BATCH_SIZE" \
    --num_rounds     "$NUM_ROUNDS" \
    --min_agents     "$MIN_AGENTS" \
    --max_agents     "$MAX_AGENTS" \
    --seed           "$SEED"

echo "Finished at: $(date)"
