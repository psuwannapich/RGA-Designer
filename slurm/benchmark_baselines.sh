#!/bin/bash
#SBATCH --job-name=arg_baselines
#SBATCH --output=logs/baselines_%A_%a.out
#SBATCH --error=logs/baselines_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --array=0-47        # 8 methods × 6 datasets = 48 tasks
#SBATCH -p gpu
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# Baseline benchmark — runs all fixed-topology comparison methods from the
# paper across all 6 datasets in a flat Slurm array.
#
# Array layout  (TASK_ID = method_idx * 6 + dataset_idx):
#
#   Methods (index 0-7):
#     0  vanilla           Direct LLM call, no agent framework, no few-shot
#     1  cot               Chain-of-Thought (single agent with few-shot prompts)
#     2  self_consistency  Self-Consistency (5 CoT samples, majority vote)
#     3  chain             Linear chain of N agents
#     4  complete          Fully-connected / Complete Graph
#     5  random            Random Graph
#     6  star              Star topology
#     7  llm_debate        LLM-Debate (fully-connected + debate prompting)
#
#   Datasets (index 0-5):
#     0  gsm8k
#     1  aqua
#     2  multiarith
#     3  svamp
#     4  humaneval
#     5  mmlu
#
# Optional env vars:
#   HF_MODEL        HuggingFace model ID  (default: Qwen/Qwen3-8B)
#   NUM_AGENTS      agents per graph      (default: per-method default)
#   SC_SAMPLES      SC sample count       (default: 5)
#   LIMIT           cap test samples      (default: all)
#   LLM_TIMEOUT     seconds per LLM call  (default: 600)
#   RESULTS_ROOT    output root dir       (default: benchmark_results)
#
# Usage:
#   sbatch slurm/benchmark_baselines.sh
#   sbatch --array=0-5  slurm/benchmark_baselines.sh   # CoT only, all datasets
#   HF_MODEL=meta-llama/Llama-3.2-3B-Instruct sbatch slurm/benchmark_baselines.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# Load CUDA modules so libcudnn.so is on LD_LIBRARY_PATH before torch imports.
_SLURM_DIR="${SLURM_SUBMIT_DIR:+${SLURM_SUBMIT_DIR}/slurm}"
source "${_SLURM_DIR:-$(dirname "${BASH_SOURCE[0]}")}/setup_cuda.sh"

# ---- Method / dataset registries -------------------------------------------
METHODS=(vanilla cot self_consistency chain complete random star llm_debate)
DATASETS=(gsm8k aqua multiarith svamp humaneval mmlu)
DATASET_JSONS=(
    "benchmark_datasets/gsm8k/gsm8k.jsonl"
    "benchmark_datasets/AQuA/AQuA.jsonl"
    "benchmark_datasets/MultiArith/MultiArith.json"
    "benchmark_datasets/SVAMP/SVAMP.json"
    "benchmark_datasets/humaneval/humaneval-py.jsonl"
    "benchmark_datasets/MMLU/data"
)

NUM_METHODS=${#METHODS[@]}
NUM_DATASETS=${#DATASETS[@]}

METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / NUM_DATASETS ))
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID % NUM_DATASETS ))

METHOD="${METHODS[$METHOD_IDX]}"
DATASET="${DATASETS[$DATASET_IDX]}"

# ---- Project root -----------------------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT_ROOT/logs"

# Ensure project root is on PYTHONPATH so `from datasets.xxx` works everywhere.
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---- Configurable knobs -----------------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
MODEL_SLUG="${HF_MODEL//\//-}"                     # Qwen/Qwen3-8B → Qwen-Qwen3-8B
USE_VLLM="${USE_VLLM:-1}"                          # 1 = vLLM backend (faster), 0 = HuggingFace
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"   # match --gres=gpu:2
DISABLE_THINKING="${DISABLE_THINKING:-1}"          # 1 = skip <think> chain (Qwen3 no-thinking mode)
export USE_VLLM VLLM_TENSOR_PARALLEL_SIZE DISABLE_THINKING PYTHONPATH
NUM_AGENTS="${NUM_AGENTS:-}"          # empty = use per-method default
SC_SAMPLES="${SC_SAMPLES:-5}"
LIMIT="${LIMIT:-}"                    # empty = evaluate all
LLM_TIMEOUT="${LLM_TIMEOUT:-600}"
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_results}"
# vanilla/cot: 8 concurrent tasks → HF batcher generates batch=8 → high GPU util
# multi-agent methods: lower value to avoid OOM from many concurrent LLM calls
case "$METHOD" in
    vanilla|cot|self_consistency) EVAL_BATCH="${EVAL_BATCH:-4}" ;;
    *)                            EVAL_BATCH="${EVAL_BATCH:-1}" ;;
esac

# ---- Build output paths -----------------------------------------------------
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/${METHOD}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/${DATASET}_${TIMESTAMP}.jsonl"
SUMMARY_LOG="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/summary.jsonl"

DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$DATASET_IDX]}"

echo "========================================"
echo "Job ID        : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node          : $SLURM_NODELIST"
echo "Method        : $METHOD  (index $METHOD_IDX)"
echo "Dataset       : $DATASET  (index $DATASET_IDX)"
echo "HF Model      : $HF_MODEL"
echo "Eval batch    : $EVAL_BATCH"
echo "Output file   : $OUTPUT_FILE"
echo "Started at    : $(date)"
echo "========================================"

cd "$PROJECT_ROOT"

# ---- Build optional flags ---------------------------------------------------
AGENTS_FLAG=""
[[ -n "$NUM_AGENTS" ]] && AGENTS_FLAG="--num_agents $NUM_AGENTS"

LIMIT_FLAG=""
[[ -n "$LIMIT" ]] && LIMIT_FLAG="--limit $LIMIT"

# ---- Run evaluation ---------------------------------------------------------
uv run baseline \
    --dataset         "$DATASET" \
    --dataset_json    "$DATASET_JSON" \
    --method          "$METHOD" \
    --llm_name        "$HF_MODEL" \
    --sc_samples      "$SC_SAMPLES" \
    --eval_batch_size "$EVAL_BATCH" \
    --timeout         "$LLM_TIMEOUT" \
    --output_file     "$OUTPUT_FILE" \
    --summary_log_file "$SUMMARY_LOG" \
    $AGENTS_FLAG \
    $LIMIT_FLAG

echo "========================================"
echo "Baseline benchmark complete: $METHOD / $DATASET"
echo "Finished at: $(date)"
echo "========================================"
