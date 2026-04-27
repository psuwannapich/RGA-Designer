#!/bin/bash
#SBATCH --job-name=arg_train
#SBATCH --output=logs/arg_%A_%a.out
#SBATCH --error=logs/arg_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5          # 0=gsm8k 1=aqua 2=multiarith 3=svamp 4=humaneval 5=mmlu
#SBATCH -p gpu
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# ARGDesigner training pipeline — one run per LLM × dataset × run index.
#
# Artifacts are saved under  <project>/<llm_slug>/arg/<run>/
# Run multiple times (RUN=0..4) to get variance estimates.
# RLHF experiments read from this directory via ARG_RUN=<run>.
#
# Stages (sequential within each array task):
#   1  cold-start    generate training graphs via LLM
#   2  pretrain      train ARGDesigner GNN on cold-start data
#   3a build_deff    build D_eff (LLM inference: D_pruned + D_simple)
#   3b nn_finetune   Phase-2 fine-tune ARGDesigner on D_eff
#   4a gen_graphs    sample graphs from the fine-tuned policy
#   4b benchmark     evaluate graphs with the LLM
#
# Array index → dataset:
#   0 gsm8k  1 aqua  2 multiarith  3 svamp  4 humaneval  5 mmlu
#
# Usage:
#   sbatch slurm/pipeline_arg.sh                      # all datasets
#   sbatch --array=0 slurm/pipeline_arg.sh            # gsm8k only
#
# Optional env vars:
#   HF_MODEL          HuggingFace model ID              (default: Qwen/Qwen3-4B)
#   LLM_SLUG          directory name override           (default: HF_MODEL with / → -)
#   RUN               repetition index 0-4              (default: 0)
#   DISABLE_THINKING  1 = no-thinking mode (Qwen3)      (default: 1)
#   NUM_TASKS         cold-start tasks (0 = all)         (default: 0)
#   NUM_ITERATIONS    cold-start iterations              (default: 10)
#   EPOCHS            pre-train epochs                   (default: 30)
#   FINETUNE_EPOCHS   Phase-2 fine-tune epochs           (default: 30)
#   FINETUNE_LR       Phase-2 learning rate              (default: 5e-5)
#   EVAL_BATCH        benchmark batch size               (default: 16)
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
TASK_SPLIT_PATHS=(
    "benchmark_datasets/gsm8k/task_split_gsm8k.json"
    "benchmark_datasets/AQuA/task_split_aqua.json"
    "benchmark_datasets/MultiArith/task_split_multiarith.json"
    "benchmark_datasets/SVAMP/task_split_svamp.json"
    "benchmark_datasets/humaneval/task_split_humaneval.json"
    "benchmark_datasets/MMLU/task_split_mmlu.json"
)
DATASET_MIN_AGENTS=(2 2 2 2 2 2)
DATASET_MAX_AGENTS=(4 4 4 4 5 6)
DECISION_METHODS=(FinalRefer FinalRefer FinalRefer FinalRefer FinalWriteCode FinalRefer)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$SLURM_ARRAY_TASK_ID]}"
TASK_SPLIT_RAW="${TASK_SPLIT_PATHS[$SLURM_ARRAY_TASK_ID]}"
MIN_AGENTS="${MIN_AGENTS:-${DATASET_MIN_AGENTS[$SLURM_ARRAY_TASK_ID]}}"
MAX_AGENTS="${MAX_AGENTS:-${DATASET_MAX_AGENTS[$SLURM_ARRAY_TASK_ID]}}"
DECISION="${DECISION_METHODS[$SLURM_ARRAY_TASK_ID]}"

# ---- Model configuration ----------------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"

if [[ "$HF_MODEL" == *"Llama"* ]] || [[ "$HF_MODEL" == *"llama"* ]]; then
    IS_LLAMA=1
else
    IS_LLAMA=0
fi

DISABLE_THINKING="${DISABLE_THINKING:-$([ "$IS_LLAMA" = "1" ] && echo 0 || echo 1)}"

# LLM_SLUG: clean directory name from model ID (no experiment suffixes).
# Override with LLM_SLUG=... env var to point at a custom directory.
LLM_SLUG="${LLM_SLUG:-${HF_MODEL//\//-}}"

# RUN: repetition index (0-4 for 5 runs). Each run is isolated under arg/<run>/.
RUN="${RUN:-0}"

export DISABLE_THINKING LLM_SLUG PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---- Hyperparameters --------------------------------------------------------
NUM_TASKS="${NUM_TASKS:-0}"
NUM_ITERATIONS="${NUM_ITERATIONS:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_ROUNDS="${NUM_ROUNDS:-1}"
SEED="${SEED:-42}"

EPOCHS="${EPOCHS:-30}"
TRAIN_LR="${TRAIN_LR:-1e-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
VAL_RATIO="${VAL_RATIO:-0.1}"

FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-30}"
FINETUNE_LR="${FINETUNE_LR:-5e-5}"
PRUNING_RATIO="${PRUNING_RATIO:-0.25}"
REPLAY_RATIO="${REPLAY_RATIO:-0.3}"

EVAL_BATCH="${EVAL_BATCH:-16}"
LIMIT="${LIMIT:-}"
NO_EF="${NO_EF:-0}"

# ---- Directory layout -------------------------------------------------------
#   <project>/<llm_slug>/arg/<run>/
#     ColdStartData/<dataset>/
#     checkpoints/<dataset>/
#     graphs/arg_designer/<dataset>_graphs.jsonl
#     benchmark_results/pregraph/arg_designer/<dataset>.jsonl
#     state/<dataset>/      ← .done sentinel files
ARG_BASE="$PROJECT_ROOT/${LLM_SLUG}/arg/${RUN}"
COLD_START_DIR="$ARG_BASE/ColdStartData/${DATASET}"
CHECKPOINT_DIR="$ARG_BASE/checkpoints/${DATASET}"
GRAPHS_DIR="$ARG_BASE/graphs/arg_designer"
RESULTS_DIR="$ARG_BASE/benchmark_results/pregraph/arg_designer"
STATE_DIR="$ARG_BASE/state/${DATASET}"
mkdir -p "$STATE_DIR"

# ---- vLLM inference server --------------------------------------------------
USE_VLLM_SERVER="${USE_VLLM_SERVER:-1}"
# Port range 7800-7805 for the ARG pipeline (one per dataset slot).
VLLM_PORT="${VLLM_PORT:-$((7800 + ${SLURM_ARRAY_TASK_ID:-0} + ${RUN} * 10))}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

if [[ "$IS_LLAMA" = "1" ]]; then
    VLLM_SERVE_DIR="${VLLM_SERVE_DIR:-/home/users/psuwannapichat/work_space/llama_serve}"
    VLLM_DTYPE="${VLLM_DTYPE:-float16}"
    VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-}"
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
        --gpu-memory-utilization 0.90
        --tensor-parallel-size   "$VLLM_TP"
        --enforce-eager)
    [[ -n "${VLLM_CHAT_TEMPLATE:-}" ]] && _vllm_cmd+=(--chat-template "$VLLM_CHAT_TEMPLATE")
    ("${_vllm_cmd[@]}") \
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
    [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true
    echo "  ✓ vLLM server stopped (EXIT trap)."
}
trap _stop_vllm_server EXIT

_ensure_vllm_running() {
    if [[ "$USE_VLLM_SERVER" != "1" ]]; then return 0; fi
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "  vLLM server already running (PID ${VLLM_PID})."
        return 0
    fi
    _start_vllm_server
}

_ensure_vllm_stopped() {
    if [[ "$USE_VLLM_SERVER" != "1" ]]; then return 0; fi
    if [[ -z "${VLLM_PID:-}" ]] || ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "  vLLM server not running — nothing to stop."
        return 0
    fi
    echo "▶ Stopping vLLM server (PID ${VLLM_PID}) to free GPU VRAM ..."
    kill "${VLLM_PID}" 2>/dev/null || true
    local _waited=0
    while kill -0 "${VLLM_PID}" 2>/dev/null; do
        sleep 2; _waited=$(( _waited + 2 ))
        if (( _waited >= 60 )); then
            echo "  WARNING: vLLM still alive after 60s — sending SIGKILL ..."
            kill -9 "${VLLM_PID}" 2>/dev/null || true
            break
        fi
    done
    VLLM_PID=""
    echo "  ✓ vLLM server stopped. Waiting 5s for GPU VRAM to be reclaimed ..."
    sleep 5
}

# ---- Stage checkpointing ----------------------------------------------------
# Delete a .done file to re-run that stage:
#   rm <llm_slug>/arg/state/<dataset>/stage1_cold_start.done
stage_done() { [[ -f "$STATE_DIR/$1.done" ]]; }
mark_done()  { touch "$STATE_DIR/$1.done"; echo "  ✔ checkpoint: $STATE_DIR/$1.done"; }

echo "════════════════════════════════════════════════"
echo "  ARGDesigner Training Pipeline"
echo "  Job $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "  Node             : $SLURM_NODELIST"
echo "  Dataset          : $DATASET"
echo "  Model            : $HF_MODEL"
echo "  Run              : $RUN"
echo "  Output base      : ${LLM_SLUG}/arg/${RUN}/"
echo "  Cold-start tasks : ${NUM_TASKS} (0=all)  iter=$NUM_ITERATIONS"
echo "  Pre-train epochs : $EPOCHS  lr=$TRAIN_LR"
echo "  Finetune  epochs : $FINETUNE_EPOCHS  lr=$FINETUNE_LR"
echo "  Eval batch       : $EVAL_BATCH"
echo "  vLLM server      : $([ "$USE_VLLM_SERVER" = "1" ] && echo "enabled (port $VLLM_PORT, tp=$VLLM_TP)" || echo "disabled (HF backend)")"
echo "  Started at       : $(date)"
echo "════════════════════════════════════════════════"

if [[ "$USE_VLLM_SERVER" == "1" ]]; then
    export LOCAL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export LOCAL_API_KEY="EMPTY"
    export USE_VLLM_SERVER USE_VLLM=0
fi

# ---- Stage 1: Cold-start ----------------------------------------------------
echo ""
if stage_done stage1_cold_start; then
    echo "▶ [Stage 1/4] Cold-start ... SKIPPED (already done)"
else
    echo "▶ [Stage 1/4] Cold-start ..."
    _ensure_vllm_running

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
    _ensure_vllm_stopped

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
    echo "  ✓ Pre-train → $CHECKPOINT_DIR/best_model.pth"
fi

# ---- Stage 3a: Build D_eff (LLM inference) ----------------------------------
echo ""
if stage_done stage3a_build_deff; then
    echo "▶ [Stage 3a/4] Build D_eff ... SKIPPED (already done)"
else
    echo "▶ [Stage 3a/4] Build D_eff ..."
    _ensure_vllm_running

    uv run python experiment/finetune_gemma.py \
        --phase            build_deff \
        --dataset          "$DATASET" \
        --dataset_json     "$DATASET_JSON" \
        --cold_start_dir   "$COLD_START_DIR" \
        --checkpoint_dir   "$CHECKPOINT_DIR" \
        --output_dir       "$CHECKPOINT_DIR" \
        --llm_name         "$HF_MODEL" \
        --pruning_ratio    "$PRUNING_RATIO" \
        --replay_ratio     "$REPLAY_RATIO" \
        --batch_size       "$BATCH_SIZE" \
        --seed             "$SEED"

    mark_done stage3a_build_deff
    echo "  ✓ D_eff → $CHECKPOINT_DIR/FinetuneData_${DATASET}"
fi

# ---- Stage 3b: NN fine-tune -------------------------------------------------
echo ""
if stage_done stage3b_nn_finetune; then
    echo "▶ [Stage 3b/4] NN fine-tune ... SKIPPED (already done)"
else
    echo "▶ [Stage 3b/4] NN fine-tune ..."
    _ensure_vllm_stopped

    uv run python experiment/finetune_gemma.py \
        --phase            finetune \
        --dataset          "$DATASET" \
        --dataset_json     "$DATASET_JSON" \
        --cold_start_dir   "$COLD_START_DIR" \
        --checkpoint_dir   "$CHECKPOINT_DIR" \
        --output_dir       "$CHECKPOINT_DIR" \
        --llm_name         "$HF_MODEL" \
        --finetune_epochs  "$FINETUNE_EPOCHS" \
        --finetune_lr      "$FINETUNE_LR" \
        --train_batch_size "$TRAIN_BATCH_SIZE" \
        --seed             "$SEED"

    mark_done stage3b_nn_finetune
    echo "  ✓ Fine-tune → $CHECKPOINT_DIR/ef_best_model.pth"
fi

# ---- Stage 4a: Generate graphs ----------------------------------------------
echo ""
GRAPHS_FILE="$GRAPHS_DIR/${DATASET}_graphs.jsonl"
NO_EF_FLAG=""
[[ "$NO_EF" == "1" ]] && NO_EF_FLAG="--no_ef"

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
        --model_type   arg_designer \
        ${TASK_SPLIT_RAW:+--task_split_path "$PROJECT_ROOT/$TASK_SPLIT_RAW"} \
        ${LIMIT:+--limit "$LIMIT"} \
        $NO_EF_FLAG

    mark_done stage4a_gen_graphs
    echo "  ✓ Graphs → $GRAPHS_FILE"
fi

# ---- Stage 4b: Benchmark ----------------------------------------------------
echo ""
if stage_done stage4b_benchmark; then
    echo "▶ [Stage 4b/4] Benchmark ... SKIPPED (already done)"
else
    echo "▶ [Stage 4b/4] Benchmark ..."
    _ensure_vllm_running

    OUTPUT_FILE="$RESULTS_DIR/${DATASET}.jsonl"
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
        --model_type       arg_designer \
        ${LIMIT:+--limit "$LIMIT"}

    mark_done stage4b_benchmark
    echo "  ✓ Results → $OUTPUT_FILE"
fi

# ---- Stage 5a: Generate graphs on TRAINING set (overfitting check) ----------
echo ""
TRAIN_GRAPHS_DIR="$ARG_BASE/graphs/arg_designer_train"
TRAIN_GRAPHS_FILE="$TRAIN_GRAPHS_DIR/${DATASET}_graphs.jsonl"

if stage_done stage5a_gen_graphs_train; then
    echo "▶ [Stage 5a/5] Generate train-set graphs ... SKIPPED (already done)"
else
    echo "▶ [Stage 5a/5] Generate train-set graphs (overfitting check) ..."
    mkdir -p "$TRAIN_GRAPHS_DIR"

    uv run python experiment/generate_graphs.py \
        --model_path   "$CHECKPOINT_DIR" \
        --dataset      "$DATASET" \
        --dataset_path "$DATASET_JSON" \
        --output_file  "$TRAIN_GRAPHS_FILE" \
        --model_type   arg_designer_train \
        --split_key    "base_tasks_indices,finetune_tasks_indices" \
        ${TASK_SPLIT_RAW:+--task_split_path "$PROJECT_ROOT/$TASK_SPLIT_RAW"} \
        $NO_EF_FLAG

    mark_done stage5a_gen_graphs_train
    echo "  ✓ Train graphs → $TRAIN_GRAPHS_FILE"
fi

# ---- Stage 5b: Benchmark train-set graphs ------------------------------------
echo ""
if stage_done stage5b_benchmark_train; then
    echo "▶ [Stage 5b/5] Benchmark train set ... SKIPPED (already done)"
else
    echo "▶ [Stage 5b/5] Benchmark train set ..."
    _ensure_vllm_running

    TRAIN_RESULTS_DIR="$ARG_BASE/benchmark_results/pregraph/arg_designer_train"
    TRAIN_OUTPUT_FILE="$TRAIN_RESULTS_DIR/${DATASET}.jsonl"
    TRAIN_SUMMARY_LOG="$TRAIN_RESULTS_DIR/summary.jsonl"
    mkdir -p "$TRAIN_RESULTS_DIR"

    uv run python experiment/benchmark_pregraph.py \
        --graphs_file      "$TRAIN_GRAPHS_FILE" \
        --dataset          "$DATASET" \
        --llm_name         "$HF_MODEL" \
        --decision_method  "$DECISION" \
        --output_file      "$TRAIN_OUTPUT_FILE" \
        --summary_log_file "$TRAIN_SUMMARY_LOG" \
        --eval_batch_size  "$EVAL_BATCH" \
        --model_type       arg_designer_train

    mark_done stage5b_benchmark_train
    echo "  ✓ Train results → $TRAIN_OUTPUT_FILE"
fi

echo ""
echo "════════════════════════════════════════════════"
echo "  Pipeline complete for $DATASET"
echo "  Test  results : ${LLM_SLUG}/arg/${RUN}/benchmark_results/pregraph/arg_designer/summary.jsonl"
echo "  Train results : ${LLM_SLUG}/arg/${RUN}/benchmark_results/pregraph/arg_designer_train/summary.jsonl"
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════"
