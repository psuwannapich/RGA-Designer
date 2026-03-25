#!/bin/bash
#SBATCH --job-name=arg_gen_graphs
#SBATCH --output=logs/gen_graphs_%A_%a.out
#SBATCH --error=logs/gen_graphs_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH -p gpu
#SBATCH --time=4:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# Stage 1 (split benchmark): generate inference graphs from a trained
# ARGDesigner model and save them to JSONL.
#
# The output is consumed by benchmark_pregraph.sh (Stage 2).
# Submit with a dependency to chain automatically:
#
#   GEN_JOB=$(sbatch --parsable slurm/generate_graphs.sh)
#   sbatch --dependency=afterok:$GEN_JOB slurm/benchmark_pregraph.sh
#
# Array mapping:  0=gsm8k  1=aqua  2=humaneval  3=mmlu  4=multiarith  5=svamp
#
# Optional env vars:
#   HF_MODEL          model slug used to locate the checkpoint (default: Qwen/Qwen3-8B)
#   CHECKPOINT_ROOT   sub-dir under MODEL_SLUG where models live (default: checkpoints)
#   GRAPHS_ROOT       output sub-dir for graph files (default: graphs)
#   LIMIT             cap number of test samples (default: all)
#   NO_EF             set to 1 to use best_model.pth instead of ef_best_model.pth
# ---------------------------------------------------------------------------

set -euo pipefail

_SLURM_DIR="${SLURM_SUBMIT_DIR:+${SLURM_SUBMIT_DIR}/slurm}"
source "${_SLURM_DIR:-$(dirname "${BASH_SOURCE[0]}")}/setup_cuda.sh"

# ---- Dataset registry -------------------------------------------------------
DATASETS=(gsm8k aqua humaneval mmlu multiarith svamp)

DATASET_PATHS=(
    "benchmark_datasets/gsm8k/gsm8k.jsonl"
    "benchmark_datasets/AQuA/AQuA.jsonl"
    "benchmark_datasets/humaneval/humaneval-py.jsonl"
    "benchmark_datasets/MMLU/data"          # directory for MMLU
    "benchmark_datasets/MultiArith/MultiArith.json"
    "benchmark_datasets/SVAMP/SVAMP.json"
)

# Empty string = no task split (use all samples)
TASK_SPLIT_PATHS=(
    "experiment/gsm8k/task_split_gsm8k.json"
    "experiment/aqua/task_split_aqua.json"
    "experiment/humaneval/task_split_humaneval.json"
    "experiment/mmlu/task_split_humaneval.json"
    "experiment/multiarith/task_split_humaneval.json"
    "experiment/svamp/task_split_humaneval.json"
)

# ---- Paths ------------------------------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT_ROOT/logs"

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATASET_PATH="$PROJECT_ROOT/${DATASET_PATHS[$SLURM_ARRAY_TASK_ID]}"
TASK_SPLIT_RAW="${TASK_SPLIT_PATHS[$SLURM_ARRAY_TASK_ID]}"

HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
MODEL_SLUG="${HF_MODEL//\//-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
GRAPHS_ROOT="${GRAPHS_ROOT:-graphs}"
LIMIT="${LIMIT:-}"
NO_EF="${NO_EF:-0}"

MODEL_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${CHECKPOINT_ROOT}/${DATASET}"
GRAPHS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${GRAPHS_ROOT}/${CHECKPOINT_ROOT}"
GRAPHS_FILE="$GRAPHS_DIR/${DATASET}_graphs.jsonl"

mkdir -p "$GRAPHS_DIR"

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH

echo "========================================"
echo "Job ID        : $SLURM_JOB_ID  (array $SLURM_ARRAY_TASK_ID)"
echo "Node          : $SLURM_NODELIST"
echo "Stage         : 1 — graph generation"
echo "Dataset       : $DATASET"
echo "Model dir     : $MODEL_DIR"
echo "Output graphs : $GRAPHS_FILE"
echo "Limit         : ${LIMIT:-all}"
echo "ef_best       : $([ "$NO_EF" = "1" ] && echo no || echo yes)"
echo "Started at    : $(date)"
echo "========================================"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: checkpoint directory not found: $MODEL_DIR"
    echo "  Run slurm/finetune.sh first, or set CHECKPOINT_ROOT."
    exit 1
fi

cd "$PROJECT_ROOT"

# Build optional flags
LIMIT_FLAG=""
[[ -n "$LIMIT" ]] && LIMIT_FLAG="--limit $LIMIT"

NO_EF_FLAG=""
[[ "$NO_EF" == "1" ]] && NO_EF_FLAG="--no_ef"

SPLIT_FLAG=""
[[ -n "$TASK_SPLIT_RAW" ]] && SPLIT_FLAG="--task_split_path $PROJECT_ROOT/$TASK_SPLIT_RAW"

uv run python experiment/generate_graphs.py \
    --model_path    "$MODEL_DIR" \
    --dataset       "$DATASET" \
    --dataset_path  "$DATASET_PATH" \
    --output_file   "$GRAPHS_FILE" \
    $SPLIT_FLAG \
    $LIMIT_FLAG \
    $NO_EF_FLAG

echo "========================================"
echo "Stage 1 complete for $DATASET"
echo "Graphs file : $GRAPHS_FILE"
echo "Finished at : $(date)"
echo "========================================"
