#!/bin/bash
#SBATCH --job-name=arg_pipeline_train
#SBATCH --output=logs/pipeline_train_%A_%a.out
#SBATCH --error=logs/pipeline_train_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --array=0-5          # 0=gsm8k 1=aqua 2=multiarith 3=svamp 4=humaneval 5=mmlu
#SBATCH -p gpu
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# ARGDesigner training pipeline — all stages in one job per dataset.
#
# Stages (sequential within each array task):
#   1. cold-start   generate training graphs via LLM
#   2. pretrain     train ARGDesigner GNN on cold-start data
#   3. finetune     Phase-2 fine-tuning (build D_eff + fine-tune)
#   4. benchmark    evaluate the fine-tuned model
#
# Array index → dataset:
#   0 gsm8k  1 aqua  2 multiarith  3 svamp  4 humaneval  5 mmlu
#
# Usage:
#   sbatch slurm/pipeline_train.sh
#   sbatch --array=0 slurm/pipeline_train.sh       # gsm8k only
#
# Optional env vars:
#   HF_MODEL          HuggingFace model ID            (default: Qwen/Qwen3-8B)
#   DISABLE_THINKING  1 = no-thinking mode (Qwen3)    (default: 1)
#   NUM_TASKS         cold-start tasks (0 = all)       (default: 0)
#   NUM_ITERATIONS    cold-start iterations            (default: 10)
#   EPOCHS            pre-train epochs                 (default: 100)
#   FINETUNE_EPOCHS   Phase-2 fine-tune epochs         (default: 200)
#   FINETUNE_LR       Phase-2 learning rate            (default: 5e-5)
#   EVAL_BATCH        benchmark batch size             (default: 8)
#   COLD_START_ROOT   cold-start data sub-dir          (default: ColdStartData)
#   CHECKPOINT_ROOT   checkpoint sub-dir               (default: checkpoints)
#   RESULTS_ROOT      results sub-dir                  (default: benchmark_results)
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT_ROOT/logs"
cd "$PROJECT_ROOT"

# ---- Dataset registry -------------------------------------------------------
DATASETS=(gsm8k aqua multiarith svamp humaneval mmlu)
DATASET_JSONS=(
    "benchmark_datasets/gsm8k/gsm8k.jsonl"
    "benchmark_datasets/AQuA/AQuA.jsonl"
    "benchmark_datasets/MultiArith/MultiArith.json"
    "benchmark_datasets/SVAMP/SVAMP.json"
    "benchmark_datasets/humaneval/humaneval-py.jsonl"
    "benchmark_datasets/MMLU/data"
)
DATASET_MIN_AGENTS=(3 3 3 3 3 3)
DATASET_MAX_AGENTS=(4 4 4 4 5 6)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$SLURM_ARRAY_TASK_ID]}"
MIN_AGENTS="${MIN_AGENTS:-${DATASET_MIN_AGENTS[$SLURM_ARRAY_TASK_ID]}}"
MAX_AGENTS="${MAX_AGENTS:-${DATASET_MAX_AGENTS[$SLURM_ARRAY_TASK_ID]}}"

# ---- Configuration ----------------------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
MODEL_SLUG="${HF_MODEL//\//-}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
MODEL_SLUG="${MODEL_SLUG}-$([ "${DISABLE_THINKING}" = "1" ] && echo no_thinking || echo thinking)"
export DISABLE_THINKING PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

NUM_TASKS="${NUM_TASKS:-0}"
NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_ROUNDS="${NUM_ROUNDS:-1}"
SEED="${SEED:-42}"

EPOCHS="${EPOCHS:-100}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_RATIO="${VAL_RATIO:-0.1}"

FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-200}"
FINETUNE_LR="${FINETUNE_LR:-5e-5}"
PRUNING_RATIO="${PRUNING_RATIO:-0.25}"
REPLAY_RATIO="${REPLAY_RATIO:-0.3}"

EVAL_BATCH="${EVAL_BATCH:-8}"
LIMIT="${LIMIT:-}"
NO_EF="${NO_EF:-0}"

COLD_START_ROOT="${COLD_START_ROOT:-ColdStartData}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
GRAPHS_ROOT="${GRAPHS_ROOT:-graphs}"
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_results}"

COLD_START_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${COLD_START_ROOT}/${DATASET}"
CHECKPOINT_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${CHECKPOINT_ROOT}/${DATASET}"

# ---- Stage checkpointing ------------------------------------------------
# Sentinel files live in STATE_DIR.  Delete one to re-run that stage.
#   rm ${MODEL_SLUG}/state/${DATASET}/stage1_cold_start.done
STATE_DIR="$PROJECT_ROOT/${MODEL_SLUG}/state/${DATASET}"
mkdir -p "$STATE_DIR"

stage_done()  { [[ -f "$STATE_DIR/$1.done" ]]; }
mark_done()   { touch "$STATE_DIR/$1.done"; echo "  ✔ checkpoint written: $STATE_DIR/$1.done"; }

echo "════════════════════════════════════════════════"
echo "  ARGDesigner Training Pipeline"
echo "  Job $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "  Node             : $SLURM_NODELIST"
echo "  Dataset          : $DATASET"
echo "  Model            : $HF_MODEL  (slug: $MODEL_SLUG)"
echo "  Cold-start tasks : ${NUM_TASKS} (0=all)  iter=$NUM_ITERATIONS"
echo "  Pre-train epochs : $EPOCHS  lr=$TRAIN_LR"
echo "  Finetune  epochs : $FINETUNE_EPOCHS  lr=$FINETUNE_LR"
echo "  Eval batch       : $EVAL_BATCH"
echo "  Started at       : $(date)"
echo "════════════════════════════════════════════════"

# ---- Stage 1: Cold-start ----------------------------------------------------
echo ""
if stage_done stage1_cold_start; then
    echo "▶ [Stage 1/4] Cold-start ... SKIPPED (already done)"
else
    echo "▶ [Stage 1/4] Cold-start ..."

    if [[ "$DATASET" == "mmlu" && ! -d "$DATASET_JSON/test" ]]; then
        echo "  MMLU data not found — downloading ..."
        uv run python benchmark_datasets/MMLU/download.py
    fi

    uv run cold-start \
        --dataset        "$DATASET" \
        --dataset_json   "$DATASET_JSON" \
        --llm_name       "$HF_MODEL" \
        --output_dir     "$COLD_START_DIR" \
        --num_tasks      "$NUM_TASKS" \
        --num_iterations "$NUM_ITERATIONS" \
        --batch_size     "$BATCH_SIZE" \
        --num_rounds     "$NUM_ROUNDS" \
        --min_agents     "$MIN_AGENTS" \
        --max_agents     "$MAX_AGENTS" \
        --seed           "$SEED"

    mark_done stage1_cold_start
    echo "  ✓ Cold-start → $COLD_START_DIR"
fi

# ---- Stage 2: Pre-train -----------------------------------------------------
echo ""
if stage_done stage2_pretrain; then
    echo "▶ [Stage 2/4] Pre-train ... SKIPPED (already done)"
else
    echo "▶ [Stage 2/4] Pre-train ..."

    uv run python experiment/pretrain.py \
        --dataset    "$DATASET" \
        --data_dir   "$COLD_START_DIR" \
        --output_dir "$CHECKPOINT_DIR" \
        --epochs     "$EPOCHS" \
        --lr         "$TRAIN_LR" \
        --batch_size "$TRAIN_BATCH_SIZE" \
        --val_ratio  "$VAL_RATIO" \
        --seed       "$SEED"

    mark_done stage2_pretrain
    echo "  ✓ Pre-train → $CHECKPOINT_DIR"
fi

# ---- Stage 3: Fine-tune -----------------------------------------------------
echo ""
if stage_done stage3_finetune; then
    echo "▶ [Stage 3/4] Fine-tune ... SKIPPED (already done)"
else
    echo "▶ [Stage 3/4] Fine-tune ..."

    uv run python experiment/finetune_gemma.py \
        --dataset          "$DATASET" \
        --dataset_json     "$DATASET_JSON" \
        --cold_start_dir   "$COLD_START_DIR" \
        --checkpoint_dir   "$CHECKPOINT_DIR" \
        --output_dir       "$CHECKPOINT_DIR" \
        --llm_name         "$HF_MODEL" \
        --finetune_epochs  "$FINETUNE_EPOCHS" \
        --finetune_lr      "$FINETUNE_LR" \
        --pruning_ratio    "$PRUNING_RATIO" \
        --replay_ratio     "$REPLAY_RATIO" \
        --batch_size       "$BATCH_SIZE" \
        --train_batch_size "$TRAIN_BATCH_SIZE" \
        --seed             "$SEED"

    mark_done stage3_finetune
    echo "  ✓ Fine-tune → $CHECKPOINT_DIR/ef_best_model.pth"
fi

# ---- Shared setup for stages 4a/4b ------------------------------------------
# Task-split paths (standard ordering: 0=gsm8k 1=aqua 2=multiarith 3=svamp 4=humaneval 5=mmlu)
TASK_SPLIT_PATHS=(
    "experiment/gsm8k/task_split_gsm8k.json"
    "experiment/aqua/task_split_aqua.json"
    "experiment/multiarith/task_split_humaneval.json"
    "experiment/svamp/task_split_humaneval.json"
    "experiment/humaneval/task_split_humaneval.json"
    "experiment/mmlu/task_split_humaneval.json"
)
TASK_SPLIT_RAW="${TASK_SPLIT_PATHS[$SLURM_ARRAY_TASK_ID]}"

MODEL_TYPE="arg_designer"
GRAPHS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${GRAPHS_ROOT}/${MODEL_TYPE}"
GRAPHS_FILE="$GRAPHS_DIR/${DATASET}_graphs.jsonl"

DECISION_METHODS=(FinalRefer FinalRefer FinalRefer FinalRefer FinalWriteCode FinalRefer)
DECISION="${DECISION_METHODS[$SLURM_ARRAY_TASK_ID]}"

export USE_VLLM="${USE_VLLM:-0}"
export VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
export PYTHONPATH

NO_EF_FLAG=""
[[ "$NO_EF" == "1" ]] && NO_EF_FLAG="--no_ef"

# ---- Stage 4a: Generate graphs ----------------------------------------------
echo ""
if stage_done stage4a_gen_graphs; then
    echo "▶ [Stage 4a/4] Generate graphs ... SKIPPED (already done)"
else
    echo "▶ [Stage 4a/4] Generate graphs ..."

    mkdir -p "$GRAPHS_DIR"

    uv run python experiment/generate_graphs.py \
        --model_path   "$CHECKPOINT_DIR" \
        --dataset      "$DATASET" \
        --dataset_path "$DATASET_JSON" \
        --output_file  "$GRAPHS_FILE" \
        --model_type   "$MODEL_TYPE" \
        ${TASK_SPLIT_RAW:+--task_split_path "$PROJECT_ROOT/$TASK_SPLIT_RAW"} \
        ${LIMIT:+--limit "$LIMIT"} \
        $NO_EF_FLAG

    mark_done stage4a_gen_graphs
    echo "  ✓ Graphs → $GRAPHS_FILE"
fi

# ---- Stage 4b: Benchmark pre-generated graphs --------------------------------
echo ""
if stage_done stage4b_benchmark; then
    echo "▶ [Stage 4b/4] Benchmark ... SKIPPED (already done)"
else
    echo "▶ [Stage 4b/4] Benchmark (pre-generated graphs) ..."

    RESULTS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/${MODEL_TYPE}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="$RESULTS_DIR/${DATASET}_${TIMESTAMP}.jsonl"
    SUMMARY_LOG="$RESULTS_DIR/summary.jsonl"
    mkdir -p "$RESULTS_DIR"

    uv run python experiment/benchmark_pregraph.py \
        --graphs_file      "$GRAPHS_FILE" \
        --dataset          "$DATASET" \
        --llm_name         "$HF_MODEL" \
        --decision_method  "$DECISION" \
        --output_file      "$OUTPUT_FILE" \
        --summary_log_file "$SUMMARY_LOG" \
        --eval_batch_size  "$EVAL_BATCH" \
        --model_type       "$MODEL_TYPE" \
        ${LIMIT:+--limit "$LIMIT"}

    mark_done stage4b_benchmark
    echo "  ✓ Results → $OUTPUT_FILE"
fi

echo ""
echo "════════════════════════════════════════════════"
echo "  Pipeline complete for $DATASET"
echo "  Summary : ${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/${MODEL_TYPE}/summary.jsonl"
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════"
