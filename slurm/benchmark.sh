#!/bin/bash
#SBATCH --job-name=arg_benchmark
#SBATCH --output=logs/benchmark_%A_%a.out
#SBATCH --error=logs/benchmark_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-5          # 0=gsm8k  1=aqua  2=humaneval  3=mmlu  4=multiarith  5=svamp
# #SBATCH --partition=gpu

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
# Required — set before submitting:
#   MODEL_PATH   path to trained ARGDesigner checkpoint directory
#                (produced by experiment/train_ARGDesigner.py or cold-start)
#
# Optional overrides:
#   HF_MODEL     HuggingFace model ID  (default: Qwen/Qwen3-8B)
#                OR Ollama short name  (default on local: gemma3)
#   EVAL_BATCH   parallel inference batch size (reduce if OOM)
#   LIMIT        cap number of test samples (empty = use all)
#
# Usage examples:
#   # Benchmark all datasets in parallel
#   MODEL_PATH=ColdStartData_hf_gsm8k sbatch slurm/benchmark.sh
#
#   # Single dataset (e.g. gsm8k only)
#   MODEL_PATH=ColdStartData_hf_gsm8k sbatch --array=0 slurm/benchmark.sh
#
#   # Override model
#   MODEL_PATH=my_checkpoint HF_MODEL=meta-llama/Llama-3.2-3B-Instruct \
#       sbatch slurm/benchmark.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- Required ---------------------------------------------------------------
if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "ERROR: MODEL_PATH must be set to the ARGDesigner checkpoint directory."
    echo "  e.g.: MODEL_PATH=ColdStartData_hf_gsm8k sbatch slurm/benchmark.sh"
    exit 1
fi

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
EVAL_BATCH="${EVAL_BATCH:-8}"
LIMIT="${LIMIT:-}"        # empty = evaluate all test samples
SEED="${SEED:-42}"
export HF_MODEL_CACHE="${HF_MODEL_CACHE:-$HOME/.cache/huggingface}"

# Absolute project root (directory containing this script's parent)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "$PROJECT_ROOT/logs"

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
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="$PROJECT_ROOT/logs/${DATASET}_results_${TIMESTAMP}.jsonl"
SUMMARY_LOG="$PROJECT_ROOT/logs/evaluation_summary.jsonl"

case "$DATASET" in

    gsm8k)
        TASK_SPLIT="$PROJECT_ROOT/experiment/gsm8k/task_split_gsm8k.json"
        cd "$PROJECT_ROOT/experiment/gsm8k"
        python evaluate_gsm8k.py \
            --model_path      "$PROJECT_ROOT/$MODEL_PATH" \
            --dataset_path    "$PROJECT_ROOT/datasets/gsm8k/gsm8k.jsonl" \
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
        cd "$PROJECT_ROOT/experiment/aqua"
        python evaluate_aqua.py \
            --model_path      "$PROJECT_ROOT/$MODEL_PATH" \
            --dataset_path    "$PROJECT_ROOT/datasets/AQuA/AQuA.jsonl" \
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
        cd "$PROJECT_ROOT/experiment/humaneval"
        python evaluate_humaneval.py \
            --model_path      "$PROJECT_ROOT/$MODEL_PATH" \
            --dataset_path    "$PROJECT_ROOT/datasets/humaneval/humaneval-py.jsonl" \
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
        if [[ ! -d "$PROJECT_ROOT/datasets/MMLU/data/test" ]]; then
            echo "MMLU data not found — running download script..."
            cd "$PROJECT_ROOT"
            python datasets/MMLU/download.py
            echo "MMLU download complete."
        fi

        cd "$PROJECT_ROOT/experiment/mmlu"
        LIMIT_MMLU_FLAG=""
        [[ -n "$LIMIT" ]] && LIMIT_MMLU_FLAG="--limit_questions $LIMIT"

        python evaluate_mmlu.py \
            --model_path      "$PROJECT_ROOT/$MODEL_PATH" \
            --data_dir        "$PROJECT_ROOT/datasets/MMLU/data" \
            --llm_name        "$HF_MODEL" \
            --decision_method FinalRefer \
            --summary_log_file "$SUMMARY_LOG" \
            --eval_batch_size "$EVAL_BATCH" \
            --seed            "$SEED" \
            $LIMIT_MMLU_FLAG
        ;;

    multiarith)
        cd "$PROJECT_ROOT/experiment/multiarith"
        python evaluate_multiarith.py \
            --model_path      "$PROJECT_ROOT/$MODEL_PATH" \
            --dataset_path    "$PROJECT_ROOT/datasets/MultiArith/MultiArith.json" \
            --llm_name        "$HF_MODEL" \
            --decision_method FinalRefer \
            --output_file     "$OUTPUT_FILE" \
            --summary_log_file "$SUMMARY_LOG" \
            --eval_batch_size "$EVAL_BATCH" \
            --seed            "$SEED" \
            $LIMIT_FLAG
        ;;

    svamp)
        cd "$PROJECT_ROOT/experiment/svamp"
        python evaluate_svamp.py \
            --model_path      "$PROJECT_ROOT/$MODEL_PATH" \
            --dataset_path    "$PROJECT_ROOT/datasets/SVAMP/SVAMP.json" \
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
