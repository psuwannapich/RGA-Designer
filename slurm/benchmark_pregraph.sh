#!/bin/bash
#SBATCH --job-name=arg_bench_pregraph
#SBATCH --output=logs/bench_pregraph_%A_%a.out
#SBATCH --error=logs/bench_pregraph_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH -p gpu
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# Stage 2 (split benchmark): run LLM inference on pre-generated graphs.
# Requires the graphs JSONL from generate_graphs.sh (Stage 1).
#
# Typical usage — chain after Stage 1:
#
#   GEN_JOB=$(sbatch --parsable slurm/generate_graphs.sh)
#   sbatch --dependency=afterok:$GEN_JOB \
#          --export=ALL,HF_MODEL="$HF_MODEL",... \
#          slurm/benchmark_pregraph.sh
#
# Or run independently (graphs must already exist):
#   sbatch slurm/benchmark_pregraph.sh
#
# Array mapping:  0=gsm8k  1=aqua  2=humaneval  3=mmlu  4=multiarith  5=svamp
#
# Optional env vars:
#   HF_MODEL          LLM for inference (default: Qwen/Qwen3-4B)
#   CHECKPOINT_ROOT   must match Stage 1 value (default: checkpoints)
#   MODEL_TYPE        must match Stage 1 value: arg_designer | rlhf
#                     (auto-derived from CHECKPOINT_ROOT if not set)
#   GRAPHS_ROOT       must match Stage 1 value (default: graphs)
#   RESULTS_ROOT      output sub-dir for results (default: benchmark_results)
#   EVAL_BATCH        async LLM inference batch size (default: 8)
#   DISABLE_THINKING  1 = Qwen3 no-thinking mode (default: 1)
#   USE_VLLM_SERVER   1 = start vLLM HTTP server (default: 1)
#   VLLM_PORT         port for vLLM server (default: 6789 + array task ID)
#   VLLM_TP           tensor-parallel GPUs for vLLM (default: 2)
#   VLLM_SERVE_DIR    directory with vLLM .venv (default: ~/work_space/vllm_temp)
# ---------------------------------------------------------------------------

MODEL_TYPE=rlhf_global_rm

set -euo pipefail

# ---- Dataset registry -------------------------------------------------------
DATASETS=(gsm8k aqua humaneval mmlu multiarith svamp)
DECISION_METHODS=(FinalRefer FinalRefer FinalWriteCode FinalRefer FinalRefer FinalRefer)

# ---- Paths ------------------------------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DECISION="${DECISION_METHODS[$SLURM_ARRAY_TASK_ID]}"

HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"
MODEL_SLUG="${HF_MODEL//\//-}"
GRAPHS_ROOT="${GRAPHS_ROOT:-graphs}"
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_results}"
EVAL_BATCH="${EVAL_BATCH:-2}"
LIMIT="${LIMIT:-}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
MODEL_SLUG="${MODEL_SLUG}-vllm-$([ "${DISABLE_THINKING}" = "1" ] && echo no_thinking || echo thinking)"

GRAPHS_FILE="$PROJECT_ROOT/${MODEL_SLUG}/${GRAPHS_ROOT}/${MODEL_TYPE}/${DATASET}_graphs.jsonl"
RESULTS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/${MODEL_TYPE}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$RESULTS_DIR/${DATASET}_${TIMESTAMP}.jsonl"
SUMMARY_LOG="$RESULTS_DIR/summary.jsonl"

mkdir -p "$RESULTS_DIR"

export DISABLE_THINKING PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---- vLLM inference server --------------------------------------------------
# Set USE_VLLM_SERVER=0 to disable and fall back to HuggingFace transformers.
USE_VLLM_SERVER="${USE_VLLM_SERVER:-1}"
VLLM_PORT="${VLLM_PORT:-$((6789 + ${SLURM_ARRAY_TASK_ID:-0}))}"
VLLM_TP="${VLLM_TP:-1}"                  # tensor-parallel GPUs for the server
VLLM_SERVE_DIR="${VLLM_SERVE_DIR:-/home/users/psuwannapichat/work_space/vllm_temp}"
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-${VLLM_SERVE_DIR}/qwen3_nonthinking.jinja}"
VLLM_PID=""

_start_vllm_server() {
    echo "▶ Starting vLLM server for '$HF_MODEL' on port $VLLM_PORT ..."
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

echo "========================================"
echo "Job ID        : $SLURM_JOB_ID  (array $SLURM_ARRAY_TASK_ID)"
echo "Node          : $SLURM_NODELIST"
echo "Stage         : 2 — LLM inference"
echo "Dataset       : $DATASET"
echo "LLM           : $HF_MODEL"
echo "Decision      : $DECISION"
echo "Graphs file   : $GRAPHS_FILE"
echo "Output file   : $OUTPUT_FILE"
echo "Eval batch    : $EVAL_BATCH"
echo "Limit         : ${LIMIT:-all}"
echo "Thinking      : $([ "$DISABLE_THINKING" = "1" ] && echo disabled || echo enabled)"
echo "vLLM server   : $([ "$USE_VLLM_SERVER" = "1" ] && echo "enabled (port $VLLM_PORT, tp=$VLLM_TP)" || echo "disabled (HF backend)")"
echo "Started at    : $(date)"
echo "========================================"

if [[ ! -f "$GRAPHS_FILE" ]]; then
    echo "ERROR: graphs file not found: $GRAPHS_FILE"
    echo "  Run slurm/generate_graphs.sh (Stage 1) first."
    echo "  Or set GRAPHS_ROOT / CHECKPOINT_ROOT to match Stage 1."
    exit 1
fi

GRAPH_COUNT=$(wc -l < "$GRAPHS_FILE")
echo "Graphs file contains $GRAPH_COUNT entries."

if [[ "$USE_VLLM_SERVER" == "1" ]]; then
    export LOCAL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export LOCAL_API_KEY="EMPTY"
    export USE_VLLM_SERVER USE_VLLM=0
    _start_vllm_server
    echo ""
fi

LIMIT_FLAG=""
[[ -n "$LIMIT" ]] && LIMIT_FLAG="--limit $LIMIT"

uv run python experiment/benchmark_pregraph.py \
    --graphs_file       "$GRAPHS_FILE" \
    --dataset           "$DATASET" \
    --llm_name          "$HF_MODEL" \
    --decision_method   "$DECISION" \
    --output_file       "$OUTPUT_FILE" \
    --summary_log_file  "$SUMMARY_LOG" \
    --eval_batch_size   "$EVAL_BATCH" \
    --model_type        "$MODEL_TYPE" \
    $LIMIT_FLAG

echo "========================================"
echo "Stage 2 complete for $DATASET"
echo "Results : $OUTPUT_FILE"
echo "Summary : $SUMMARY_LOG"
echo "Finished at : $(date)"
echo "========================================"
