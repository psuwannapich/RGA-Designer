#!/bin/bash
#SBATCH --job-name=arg_benchmark
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --array=0-5          # 0=gsm8k  1=aqua  2=humaneval  3=mmlu  4=multiarith  5=svamp
#SBATCH -p gpu
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu
# ---------------------------------------------------------------------------
# Benchmark ARG-Designer on all supported datasets using a local model.
#
# Array index → dataset:
#   0  gsm8k
#   1  aqua
#   2  humaneval
#   3  mmlu
#   4  multiarith
#   5  svamp
#
# Optional overrides:
#   CHECKPOINT_ROOT  root dir of finetune checkpoints (default: checkpoints)
#                    Per-dataset path is: $CHECKPOINT_ROOT/<dataset>/
#                    This matches the output layout of slurm/finetune.sh.
#   HF_MODEL         HuggingFace model ID  (default: Qwen/Qwen3-8B)
#   EVAL_BATCH       parallel inference batch size (reduce if OOM)
#   LIMIT            cap number of test samples (empty = use all)
#
# Usage examples:
#   # Benchmark all datasets using default checkpoint layout
#   sbatch slurm/benchmark.sh
#
#   # Custom checkpoint root
#   CHECKPOINT_ROOT=my_checkpoints sbatch slurm/benchmark.sh
#
#   # Single dataset
#   sbatch --array=0 slurm/benchmark.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# Load CUDA modules so libcudnn.so is on LD_LIBRARY_PATH before torch imports.
_SLURM_DIR="${SLURM_SUBMIT_DIR:+${SLURM_SUBMIT_DIR}/slurm}"
source "${_SLURM_DIR:-$(dirname "${BASH_SOURCE[0]}")}/setup_cuda.sh"

# ---- Dataset registry -------------------------------------------------------
DATASETS=(
    gsm8k       # 0
    aqua        # 1
    humaneval   # 2
    mmlu        # 3
    multiarith  # 4
    svamp       # 5
)
DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

# ---- Global defaults --------------------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
MODEL_SLUG="${HF_MODEL//\//-}"                     # Qwen/Qwen3-8B → Qwen-Qwen3-8B
USE_VLLM="${USE_VLLM:-1}"                          # 1 = vLLM backend (faster), 0 = HuggingFace
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"   # match --gres=gpu:2
DISABLE_THINKING="${DISABLE_THINKING:-1}"          # 1 = skip <think> chain (Qwen3 no-thinking mode)
MODEL_SLUG="${MODEL_SLUG}-$([ "${DISABLE_THINKING}" = "1" ] && echo no_thinking || echo thinking)"
export USE_VLLM VLLM_TENSOR_PARALLEL_SIZE DISABLE_THINKING PYTHONPATH
EVAL_BATCH="${EVAL_BATCH:-8}"
LIMIT="${LIMIT:-}"        # empty = evaluate all test samples
SEED="${SEED:-42}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"

# Absolute project root (directory containing this script's parent)
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

mkdir -p "$PROJECT_ROOT/logs"

# Ensure the project root is on the Python path so that `from datasets.xxx`
# imports work regardless of which experiment sub-directory the script cd's into.
PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Per-dataset checkpoint — mirrors finetune.sh output layout
MODEL_PATH="$PROJECT_ROOT/${MODEL_SLUG}/${CHECKPOINT_ROOT}/${DATASET}"

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: checkpoint directory not found: $MODEL_PATH"
    echo "  Run slurm/finetune.sh first, or set CHECKPOINT_ROOT to the correct root."
    exit 1
fi

echo "========================================"
echo "Job ID        : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node          : $SLURM_NODELIST"
echo "Dataset       : $DATASET"
echo "Model path    : $MODEL_PATH"
echo "LLM           : $HF_MODEL"
echo "Eval batch    : $EVAL_BATCH"
echo "Limit         : ${LIMIT:-all}"
echo "Started at    : $(date)"
echo "========================================"

# ---- Build optional --limit flag --------------------------------------------
LIMIT_FLAG=""
[[ -n "$LIMIT" ]] && LIMIT_FLAG="--limit $LIMIT"

# ---- Per-dataset paths and arguments ----------------------------------------
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_results}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/arg_designer"
OUTPUT_FILE="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/arg_designer/${DATASET}_${TIMESTAMP}.jsonl"
SUMMARY_LOG="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/summary.jsonl"

cd "$PROJECT_ROOT"

case "$DATASET" in

    gsm8k)
        TASK_SPLIT="$PROJECT_ROOT/experiment/gsm8k/task_split_gsm8k.json"
        uv run python experiment/gsm8k/evaluate_gsm8k.py \
            --model_path      "$MODEL_PATH" \
            --dataset_path    "$PROJECT_ROOT/benchmark_datasets/gsm8k/gsm8k.jsonl" \
            --task_split_path "$TASK_SPLIT" \
            --llm_name        "$HF_MODEL" \
            --decision_method FinalRefer \
            --output_file     "$OUTPUT_FILE" \
            --summary_log_file "$SUMMARY_LOG" \
            --eval_batch_size "$EVAL_BATCH" \
            $LIMIT_FLAG
        ;;

    aqua)
        TASK_SPLIT="$PROJECT_ROOT/experiment/aqua/task_split_aqua.json"
        uv run python experiment/aqua/evaluate_aqua.py \
            --model_path      "$MODEL_PATH" \
            --dataset_path    "$PROJECT_ROOT/benchmark_datasets/AQuA/AQuA.jsonl" \
            --task_split_path "$TASK_SPLIT" \
            --llm_name        "$HF_MODEL" \
            --decision_method FinalRefer \
            --output_file     "$OUTPUT_FILE" \
            --summary_log_file "$SUMMARY_LOG" \
            --eval_batch_size "$EVAL_BATCH" \
            $LIMIT_FLAG
        ;;

    humaneval)
        TASK_SPLIT="$PROJECT_ROOT/experiment/humaneval/task_split_humaneval.json"
        uv run python experiment/humaneval/evaluate_humaneval.py \
            --model_path      "$MODEL_PATH" \
            --dataset_path    "$PROJECT_ROOT/benchmark_datasets/humaneval/humaneval-py.jsonl" \
            --task_split_path "$TASK_SPLIT" \
            --llm_name        "$HF_MODEL" \
            --decision_method FinalWriteCode \
            --output_file     "$OUTPUT_FILE" \
            --summary_log_file "$SUMMARY_LOG" \
            --eval_batch_size "$EVAL_BATCH" \
            $LIMIT_FLAG
        ;;

    mmlu)
        # Download MMLU data if not present
        if [[ ! -d "$PROJECT_ROOT/benchmark_datasets/MMLU/data/test" ]]; then
            echo "MMLU data not found — running download script..."
            uv run python benchmark_datasets/MMLU/download.py
            echo "MMLU download complete."
        fi

        LIMIT_MMLU_FLAG=""
        [[ -n "$LIMIT" ]] && LIMIT_MMLU_FLAG="--limit_questions $LIMIT"

        uv run python experiment/mmlu/evaluate_mmlu.py \
            --model_path      "$MODEL_PATH" \
            --data_dir        "$PROJECT_ROOT/benchmark_datasets/MMLU/data" \
            --llm_name        "$HF_MODEL" \
            --decision_method FinalRefer \
            --output_file     "$OUTPUT_FILE" \
            --summary_log_file "$SUMMARY_LOG" \
            --eval_batch_size "$EVAL_BATCH" \
            --seed            "$SEED" \
            $LIMIT_MMLU_FLAG
        ;;

    multiarith)
        uv run python experiment/multiarith/evaluate_multiarith.py \
            --model_path      "$MODEL_PATH" \
            --dataset_path    "$PROJECT_ROOT/benchmark_datasets/MultiArith/MultiArith.json" \
            --llm_name        "$HF_MODEL" \
            --decision_method FinalRefer \
            --output_file     "$OUTPUT_FILE" \
            --summary_log_file "$SUMMARY_LOG" \
            --eval_batch_size "$EVAL_BATCH" \
            --seed            "$SEED" \
            $LIMIT_FLAG
        ;;

    svamp)
        uv run python experiment/svamp/evaluate_svamp.py \
            --model_path      "$MODEL_PATH" \
            --dataset_path    "$PROJECT_ROOT/benchmark_datasets/SVAMP/SVAMP.json" \
            --llm_name        "$HF_MODEL" \
            --decision_method FinalRefer \
            --output_file     "$OUTPUT_FILE" \
            --summary_log_file "$SUMMARY_LOG" \
            --eval_batch_size "$EVAL_BATCH" \
            --seed            "$SEED" \
            $LIMIT_FLAG
        ;;
esac

echo "========================================"
echo "Finished at: $(date)"
echo "Results: $OUTPUT_FILE"
echo "Summary: $SUMMARY_LOG"
echo "========================================"
