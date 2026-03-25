#!/bin/bash
#SBATCH --job-name=arg_bench_pregraph
#SBATCH --output=logs/bench_pregraph_%A_%a.out
#SBATCH --error=logs/bench_pregraph_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --array=0-5
#SBATCH -p gpu
#SBATCH --time=12:00:00
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
#   HF_MODEL                  LLM for inference (default: Qwen/Qwen3-8B)
#   CHECKPOINT_ROOT           must match Stage 1 value (default: checkpoints)
#   MODEL_TYPE                must match Stage 1 value: arg_designer | rlhf
#                             (auto-derived from CHECKPOINT_ROOT if not set)
#   GRAPHS_ROOT               must match Stage 1 value (default: graphs)
#   RESULTS_ROOT              output sub-dir for results (default: benchmark_results)
#   EVAL_BATCH                async LLM inference batch size (default: 8)
#   DISABLE_THINKING          1 = Qwen3 no-thinking mode (default: 1)
#   VLLM_TENSOR_PARALLEL_SIZE GPUs for vLLM tensor parallelism (default: 2)
# ---------------------------------------------------------------------------

set -euo pipefail

_SLURM_DIR="${SLURM_SUBMIT_DIR:+${SLURM_SUBMIT_DIR}/slurm}"
source "${_SLURM_DIR:-$(dirname "${BASH_SOURCE[0]}")}/setup_cuda.sh"

# ---- Dataset registry -------------------------------------------------------
DATASETS=(gsm8k aqua humaneval mmlu multiarith svamp)
DECISION_METHODS=(FinalRefer FinalRefer FinalWriteCode FinalRefer FinalRefer FinalRefer)

# ---- Paths ------------------------------------------------------------------
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT_ROOT/logs"

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DECISION="${DECISION_METHODS[$SLURM_ARRAY_TASK_ID]}"

HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
MODEL_SLUG="${HF_MODEL//\//-}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
GRAPHS_ROOT="${GRAPHS_ROOT:-graphs}"
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_results}"
EVAL_BATCH="${EVAL_BATCH:-8}"
LIMIT="${LIMIT:-}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
MODEL_SLUG="${MODEL_SLUG}-$([ "${DISABLE_THINKING}" = "1" ] && echo no_thinking || echo thinking)"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
USE_VLLM="${USE_VLLM:-1}"

# Auto-derive MODEL_TYPE from CHECKPOINT_ROOT if not set explicitly
if [[ -z "${MODEL_TYPE:-}" ]]; then
    [[ "$CHECKPOINT_ROOT" == *rlhf* ]] && MODEL_TYPE="rlhf" || MODEL_TYPE="arg_designer"
fi

GRAPHS_FILE="$PROJECT_ROOT/${MODEL_SLUG}/${GRAPHS_ROOT}/${MODEL_TYPE}/${DATASET}_graphs.jsonl"
RESULTS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/${MODEL_TYPE}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$RESULTS_DIR/${DATASET}_${TIMESTAMP}.jsonl"
SUMMARY_LOG="$RESULTS_DIR/summary.jsonl"

mkdir -p "$RESULTS_DIR"

PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTHONPATH DISABLE_THINKING VLLM_TENSOR_PARALLEL_SIZE USE_VLLM

echo "========================================"
echo "Job ID        : $SLURM_JOB_ID  (array $SLURM_ARRAY_TASK_ID)"
echo "Node          : $SLURM_NODELIST"
echo "Stage         : 2 — LLM inference"
echo "Dataset       : $DATASET"
echo "Graph model   : $MODEL_TYPE  (checkpoint root: $CHECKPOINT_ROOT)"
echo "LLM           : $HF_MODEL"
echo "Decision      : $DECISION"
echo "Graphs file   : $GRAPHS_FILE"
echo "Output file   : $OUTPUT_FILE"
echo "Eval batch    : $EVAL_BATCH"
echo "Limit         : ${LIMIT:-all}"
echo "Thinking      : $([ "$DISABLE_THINKING" = "1" ] && echo disabled || echo enabled)"
echo "vLLM TP size  : $VLLM_TENSOR_PARALLEL_SIZE"
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

cd "$PROJECT_ROOT"

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
