#!/bin/bash
#SBATCH --job-name=arg_baselines_multirun
#SBATCH --output=logs/baselines_multirun_%A_%a.out
#SBATCH --error=logs/baselines_multirun_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-479       # 10 runs × 8 methods × 6 datasets = 480 tasks
#SBATCH -p gpu
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# Multi-run baseline benchmark — runs all fixed-topology comparison methods
# across all 6 datasets for 10 independent repetitions.
#
# Array layout:
#   SLURM_ARRAY_TASK_ID = method_idx * 60 + dataset_idx * 10 + run_idx
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
# Output structure:
#   BASE_PATH/<method_name>/<run_number>/<dataset>.jsonl
#   BASE_PATH/<method_name>/<run_number>/summary.jsonl
#
# Optional env vars:
#   HF_MODEL        HuggingFace model ID  (default: Qwen/Qwen3-4B)
#   NUM_AGENTS      agents per graph      (default: per-method default)
#   SC_SAMPLES      SC sample count       (default: 5)
#   LIMIT           cap test samples      (default: all)
#   BASE_PATH       root output directory (default: /mnt/scratch/users/psuwannapichat/RLHF-Designer/experiments)
#
# Usage:
#   sbatch slurm/benchmark_baselines_multirun.sh
#   sbatch --array=0-47  slurm/benchmark_baselines_multirun.sh   # run 1 only
#   sbatch --array=0,6,12,18,24,30,36,42 slurm/benchmark_baselines_multirun.sh  # vanilla only
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

NUM_METHODS=${#METHODS[@]}     # 8
NUM_DATASETS=${#DATASETS[@]}   # 6
NUM_RUNS=10

# ---- Decode array task ID --------------------------------------------------
# Layout: task_id = method_idx * (NUM_DATASETS * NUM_RUNS) + dataset_idx * NUM_RUNS + run_idx
METHOD_IDX=$(( SLURM_ARRAY_TASK_ID / (NUM_DATASETS * NUM_RUNS) ))
REMAINDER=$(( SLURM_ARRAY_TASK_ID % (NUM_DATASETS * NUM_RUNS) ))
DATASET_IDX=$(( REMAINDER / NUM_RUNS ))
RUN_IDX=$(( REMAINDER % NUM_RUNS ))

RUN_NUMBER=$(( RUN_IDX + 1 ))   # 1-indexed for human-readable directory names
METHOD="${METHODS[$METHOD_IDX]}"
DATASET="${DATASETS[$DATASET_IDX]}"

# ---- Project root -----------------------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT_ROOT/logs"

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH

# ---- Configurable knobs -----------------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"

# ---- Model-family detection -------------------------------------------------
# Automatically selects the correct vLLM venv, dtype, and chat template.
if [[ "$HF_MODEL" == *"Llama"* ]] || [[ "$HF_MODEL" == *"llama"* ]]; then
    IS_LLAMA=1
else
    IS_LLAMA=0
fi

# Thinking mode is Qwen3-only; disable unconditionally for Llama.
DISABLE_THINKING="${DISABLE_THINKING:-$([ "$IS_LLAMA" = "1" ] && echo 0 || echo 1)}"
export DISABLE_THINKING

# ---- Output base path -------------------------------------------------------
BASE_PATH="${BASE_PATH:-/mnt/scratch/users/psuwannapichat/RLHF-Designer/experiments}"
OUTPUT_DIR="${BASE_PATH}/${METHOD}/${RUN_NUMBER}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="${OUTPUT_DIR}/${DATASET}.jsonl"
SUMMARY_LOG="${OUTPUT_DIR}/summary.jsonl"

# ---- vLLM inference server --------------------------------------------------
USE_VLLM_SERVER="${USE_VLLM_SERVER:-1}"
# Each task gets a unique port derived from its full task ID to avoid collisions
VLLM_PORT="${VLLM_PORT:-$((7000 + SLURM_ARRAY_TASK_ID))}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
# Llama (GPTQ) uses llama_serve venv + dtype=auto + built-in chat template.
# Qwen3 uses vllm_serve venv + dtype=float16 + nonthinking Jinja override.
if [[ "$IS_LLAMA" = "1" ]]; then
    VLLM_SERVE_DIR="${VLLM_SERVE_DIR:-/home/users/psuwannapichat/work_space/llama_serve}"
    VLLM_DTYPE="${VLLM_DTYPE:-auto}"
    VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-}"   # use model's built-in Llama template
else
    VLLM_SERVE_DIR="${VLLM_SERVE_DIR:-/home/users/psuwannapichat/work_space/vllm_serve}"
    VLLM_DTYPE="${VLLM_DTYPE:-float16}"
    VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-${VLLM_SERVE_DIR}/qwen3_nonthinking.jinja}"
fi
VLLM_PID=""

_start_vllm_server() {
    echo "▶ Starting vLLM server for '$HF_MODEL' on port $VLLM_PORT ..."
    mkdir -p "$PROJECT_ROOT/logs"
    local _vllm_cmd=("$VLLM_SERVE_DIR/.venv/bin/vllm" serve "$HF_MODEL"
        --port                   "$VLLM_PORT"
        --dtype                  "$VLLM_DTYPE"
        --trust-remote-code
        --max-model-len          "$VLLM_MAX_MODEL_LEN"
        --gpu-memory-utilization 0.9
        --tensor-parallel-size   "$VLLM_TP"
        --enforce-eager)
    # Only pass --chat-template when a custom template is specified (Qwen3).
    # Llama Instruct models use their built-in template.
    [[ -n "${VLLM_CHAT_TEMPLATE:-}" ]] && _vllm_cmd+=(--chat-template "$VLLM_CHAT_TEMPLATE")
    ("${_vllm_cmd[@]}") \
        > "$PROJECT_ROOT/logs/vllm_${SLURM_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}.log" 2>&1 &
    VLLM_PID=$!
    echo "  Server PID : $VLLM_PID"
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

# ---- Per-method eval batch size --------------------------------------------
NUM_AGENTS="${NUM_AGENTS:-}"
SC_SAMPLES="${SC_SAMPLES:-5}"
LIMIT="${LIMIT:-}"

case "$METHOD" in
    vanilla|cot|self_consistency) EVAL_BATCH="${EVAL_BATCH:-4}" ;;
    *)                            EVAL_BATCH="${EVAL_BATCH:-1}" ;;
esac

DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$DATASET_IDX]}"
TASK_SPLIT="$PROJECT_ROOT/${TASK_SPLIT_PATHS[$DATASET_IDX]}"

echo "========================================"
echo "Job ID        : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node          : $SLURM_NODELIST"
echo "Run           : $RUN_NUMBER / 10"
echo "Method        : $METHOD  (index $METHOD_IDX)"
echo "Dataset       : $DATASET  (index $DATASET_IDX)"
echo "HF Model      : $HF_MODEL"
echo "Eval batch    : $EVAL_BATCH"
echo "Output file   : $OUTPUT_FILE"
echo "Summary log   : $SUMMARY_LOG"
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
echo "Baseline benchmark complete: $METHOD / $DATASET (run $RUN_NUMBER)"
echo "Finished at: $(date)"
echo "========================================"
