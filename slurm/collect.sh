#!/bin/bash
#SBATCH --job-name=arg_collect
#SBATCH --output=logs/collect_%A_%a.out
#SBATCH --error=logs/collect_%A_%a.err
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
# RLHF Phase 1 — collect preference pairs for all 6 datasets in parallel.
#
# Array index → dataset:
#   0  gsm8k
#   1  aqua
#   2  multiarith
#   3  svamp
#   4  humaneval
#   5  mmlu
#
# Optional env vars:
#   HF_MODEL                  HuggingFace model ID    (default: Qwen/Qwen3-8B)
#   NUM_TASKS                 tasks to sample         (default: 100)
#   NUM_GPUS                  GPUs per job            (default: 2)
#   PREFERENCE_ROOT           root dir for output     (default: rlhf_data)
#   DATASETS_ARRAY            Slurm array spec        (default: 0-5)
#   ARG_MODEL_DIR             trained ARGDesigner checkpoint dir for model-based
#                             candidate generation (optional; uses default configs only if unset)
#   SAMPLE_TEMPERATURES       space-separated temperature list for diversity sampling
#                             (default: "1.0 1.5 2.0"; higher T → more structural variety)
#   COLDSTART_DIRS            space-separated .pt directories to convert without LLM re-inference
#                             e.g. "MODEL/ColdStartData/gsm8k MODEL/ColdStartData/gsm8k_ef"
#   CHECKPOINT_ROOT           checkpoint sub-dir (default: checkpoints)
#
# Usage:
#   sbatch slurm/collect.sh
#   DATASETS_ARRAY=0 sbatch slurm/collect.sh   # gsm8k only
# ---------------------------------------------------------------------------

set -euo pipefail

# Load CUDA modules so libcudnn.so is on LD_LIBRARY_PATH before torch imports.
_SLURM_DIR="${SLURM_SUBMIT_DIR:+${SLURM_SUBMIT_DIR}/slurm}"
source "${_SLURM_DIR:-$(dirname "${BASH_SOURCE[0]}")}/setup_cuda.sh"

DATASETS=(gsm8k aqua multiarith svamp humaneval mmlu)
DATASET_JSONS=(
    "benchmark_datasets/gsm8k/gsm8k.jsonl"
    "benchmark_datasets/AQuA/AQuA.jsonl"
    "benchmark_datasets/MultiArith/MultiArith.json"
    "benchmark_datasets/SVAMP/SVAMP.json"
    "benchmark_datasets/humaneval/humaneval-py.jsonl"
    "benchmark_datasets/MMLU/data"
)
DATASET_MAX_AGENTS=(4 4 4 4 5 6)

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT_ROOT/logs"

# Ensure project root is on PYTHONPATH so `from datasets.xxx` works everywhere.
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$SLURM_ARRAY_TASK_ID]}"

HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
MODEL_SLUG="${HF_MODEL//\//-}"                     # Qwen/Qwen3-8B → Qwen-Qwen3-8B
USE_VLLM="${USE_VLLM:-1}"                          # 1 = vLLM backend (faster), 0 = HuggingFace
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"   # match --gres=gpu:2
DISABLE_THINKING="${DISABLE_THINKING:-1}"          # 1 = skip <think> chain (Qwen3 no-thinking mode)
export USE_VLLM VLLM_TENSOR_PARALLEL_SIZE DISABLE_THINKING PYTHONPATH
NUM_TASKS="${NUM_TASKS:-100}"
PREFERENCE_ROOT="${PREFERENCE_ROOT:-rlhf_data}"
PREFERENCE_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${PREFERENCE_ROOT}/${DATASET}"
MIN_AGENTS="${MIN_AGENTS:-2}"
MAX_AGENTS="${MAX_AGENTS:-${DATASET_MAX_AGENTS[$SLURM_ARRAY_TASK_ID]}}"
W_CORRECT="${W_CORRECT:-0.6}"
W_SIZE="${W_SIZE:-0.2}"
W_TOKEN="${W_TOKEN:-0.2}"
PAIR_MARGIN="${PAIR_MARGIN:-0.05}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2}"
LLM_TIMEOUT="${LLM_TIMEOUT:-600}"
SEED="${SEED:-42}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
ARG_MODEL_DIR="${ARG_MODEL_DIR:-}"         # optional: path to trained ARGDesigner checkpoint
SAMPLE_TEMPERATURES="${SAMPLE_TEMPERATURES:-1.0 1.5 2.0}"  # space-separated temperature list
COLDSTART_DIRS="${COLDSTART_DIRS:-}"      # optional: space-separated .pt directories (no LLM re-inference)

echo "========================================"
echo "Job ID        : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node          : $SLURM_NODELIST"
echo "Dataset       : $DATASET"
echo "HF Model      : $HF_MODEL"
echo "Num tasks     : $NUM_TASKS"
echo "Max agents    : $MAX_AGENTS"
echo "Preference dir: $PREFERENCE_DIR"
echo "ARG model dir : ${ARG_MODEL_DIR:-none (default configs only)}"
echo "Sample temps  : $SAMPLE_TEMPERATURES"
echo "Started at    : $(date)"
echo "========================================"

cd "$PROJECT_ROOT"

uv run rlhf \
    --dataset        "$DATASET" \
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
    --llm_timeout    "$LLM_TIMEOUT" \
    --seed           "$SEED" \
    --sample_temperatures $SAMPLE_TEMPERATURES \
    ${ARG_MODEL_DIR:+--arg_model_dir "$ARG_MODEL_DIR"} \
    ${COLDSTART_DIRS:+--coldstart_dirs $COLDSTART_DIRS}

echo "========================================"
echo "Collect complete for $DATASET"
echo "Finished at: $(date)"
echo "========================================"
