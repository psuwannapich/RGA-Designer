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
#   RESULTS_ROOT    output root dir       (default: benchmark_results)
#
# Usage:
#   sbatch slurm/benchmark_baselines.sh
#   sbatch --array=0-5  slurm/benchmark_baselines.sh   # CoT only, all datasets
#   HF_MODEL=meta-llama/Llama-3.2-3B-Instruct sbatch slurm/benchmark_baselines.sh
# ---------------------------------------------------------------------------

set -euo pipefail

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
TASK_SPLIT_PATHS=(
    "benchmark_datasets/gsm8k/task_split_gsm8k.json"
    "benchmark_datasets/AQuA/task_split_aqua.json"
    "benchmark_datasets/MultiArith/task_split_multiarith.json"
    "benchmark_datasets/SVAMP/task_split_svamp.json"
    "benchmark_datasets/humaneval/task_split_humaneval.json"
    "benchmark_datasets/MMLU/task_split_mmlu.json"
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
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"
MODEL_SLUG="${HF_MODEL//\//-}"                     # Qwen/Qwen3-8B → Qwen-Qwen3-8B
DISABLE_THINKING="${DISABLE_THINKING:-1}"          # 1 = skip <think> chain (Qwen3 no-thinking mode)
MODEL_SLUG="${MODEL_SLUG}-$([ "${DISABLE_THINKING}" = "1" ] && echo no_thinking || echo thinking)"
export DISABLE_THINKING PYTHONPATH

# ---- vLLM inference server --------------------------------------------------
# Set USE_VLLM_SERVER=0 to disable and fall back to HuggingFace transformers.
USE_VLLM_SERVER="${USE_VLLM_SERVER:-1}"
VLLM_PORT="${VLLM_PORT:-$((6789 + ${SLURM_ARRAY_TASK_ID:-0}))}"
VLLM_TP="${VLLM_TP:-2}"                  # tensor-parallel GPUs for the server
VLLM_SERVE_DIR="${VLLM_SERVE_DIR:-/home/users/psuwannapichat/work_space/vllm_temp}"
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-${VLLM_SERVE_DIR}/qwen3_nonthinking.jinja}"
VLLM_PID=""

_start_vllm_server() {
    echo "▶ Starting vLLM server for '$HF_MODEL' on port $VLLM_PORT ..."
    mkdir -p "$PROJECT_ROOT/logs"
    ("$VLLM_SERVE_DIR/.venv/bin/vllm" serve "$HF_MODEL" \
        --port                   "$VLLM_PORT" \
        --dtype                  float16 \
        --trust-remote-code \
        --max-model-len          8192 \
        --gpu-memory-utilization 0.8 \
        --tensor-parallel-size   "$VLLM_TP" \
        --enforce-eager \
        --chat-template          "$VLLM_CHAT_TEMPLATE") \
        > "$PROJECT_ROOT/logs/vllm_${SLURM_JOB_ID:-local}.log" 2>&1 &
    VLLM_PID=$!
    echo "  Server PID : $VLLM_PID"
    echo "  Server log : logs/vllm_${SLURM_JOB_ID:-local}.log"
    echo "  Waiting for vLLM to be ready ..."
    for _i in $(seq 1 120); do
        if curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
            echo "  ✓ vLLM server ready (waited $((_i * 5))s)"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: vLLM server did not start within 10 minutes." >&2
    exit 1
}

_stop_vllm_server() {
    [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null && echo "  ✓ vLLM server stopped."
}
trap _stop_vllm_server EXIT
NUM_AGENTS="${NUM_AGENTS:-}"          # empty = use per-method default
SC_SAMPLES="${SC_SAMPLES:-5}"
LIMIT="${LIMIT:-}"                    # empty = evaluate all
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
OUTPUT_FILE="$OUTPUT_DIR/${DATASET}.jsonl"
SUMMARY_LOG="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/summary.jsonl"

DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$DATASET_IDX]}"
TASK_SPLIT="$PROJECT_ROOT/${TASK_SPLIT_PATHS[$DATASET_IDX]}"

echo "========================================"
echo "Job ID        : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node          : $SLURM_NODELIST"
echo "Method        : $METHOD  (index $METHOD_IDX)"
echo "Dataset       : $DATASET  (index $DATASET_IDX)"
echo "HF Model      : $HF_MODEL"
echo "Eval batch    : $EVAL_BATCH"
echo "Output file   : $OUTPUT_FILE"
echo "vLLM server   : $([ "$USE_VLLM_SERVER" = "1" ] && echo "enabled (port $VLLM_PORT, tp=$VLLM_TP)" || echo "disabled (HF backend)")"
echo "Started at    : $(date)"
echo "========================================"

cd "$PROJECT_ROOT"

if [[ "$USE_VLLM_SERVER" == "1" ]]; then
    export LOCAL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export LOCAL_API_KEY="EMPTY"
    export USE_VLLM_SERVER USE_VLLM=0
    _start_vllm_server
    echo ""
fi

# ---- Build optional flags ---------------------------------------------------
AGENTS_FLAG=""
[[ -n "$NUM_AGENTS" ]] && AGENTS_FLAG="--num_agents $NUM_AGENTS"

LIMIT_FLAG=""
[[ -n "$LIMIT" ]] && LIMIT_FLAG="--limit $LIMIT"

SPLIT_FLAG=""
[[ -f "$TASK_SPLIT" ]] && SPLIT_FLAG="--task_split_path $TASK_SPLIT"

# ---- Run evaluation ---------------------------------------------------------
uv run baseline \
    --dataset         "$DATASET" \
    --dataset_json    "$DATASET_JSON" \
    --method          "$METHOD" \
    --llm_name        "$HF_MODEL" \
    --sc_samples      "$SC_SAMPLES" \
    --eval_batch_size "$EVAL_BATCH" \
    --output_file     "$OUTPUT_FILE" \
    --summary_log_file "$SUMMARY_LOG" \
    $AGENTS_FLAG \
    $LIMIT_FLAG \
    $SPLIT_FLAG

echo "========================================"
echo "Baseline benchmark complete: $METHOD / $DATASET"
echo "Finished at: $(date)"
echo "========================================"
