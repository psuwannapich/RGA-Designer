#!/bin/bash
#SBATCH --job-name=arg_qwen4_train
#SBATCH --output=logs/qwen4_train_%A_%a.out
#SBATCH --error=logs/qwen4_train_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5          # 0=gsm8k 1=aqua 2=multiarith 3=svamp 4=humaneval 5=mmlu
#SBATCH -p gpu
#SBATCH --time=6:00:00
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
#   RUN_NUM           run number suffix appended to MODEL_SLUG (e.g. 1 → slug-run1)
#                     also offsets VLLM_PORT by RUN_NUM*100 to prevent collisions
#                     (default: empty — no suffix, no offset)
#   NUM_TASKS         cold-start tasks (0 = all)       (default: 0)
#   NUM_ITERATIONS    cold-start iterations            (default: 10)
#   EPOCHS            pre-train epochs                 (default: 100)
#   FINETUNE_EPOCHS   Phase-2 fine-tune epochs         (default: 200)
#   FINETUNE_LR       Phase-2 learning rate            (default: 5e-5)
#   EVAL_BATCH        benchmark batch size             (default: 8)
#   COLD_START_ROOT   cold-start data sub-dir          (default: ColdStartData)
#   CHECKPOINT_ROOT   checkpoint sub-dir               (default: checkpoints)
#   RESULTS_ROOT      results sub-dir                  (default: benchmark_results)
#   AUTO_SUBMIT_RLHF  1 = auto-submit pipeline_rlhf.sh after stage3_finetune (default: 1)
#                     set to 0 to disable
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
DATASET_MIN_AGENTS=(2 2 2 2 2 2)
DATASET_MAX_AGENTS=(4 4 4 4 5 6)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$SLURM_ARRAY_TASK_ID]}"
MIN_AGENTS="${MIN_AGENTS:-${DATASET_MIN_AGENTS[$SLURM_ARRAY_TASK_ID]}}"
MAX_AGENTS="${MAX_AGENTS:-${DATASET_MAX_AGENTS[$SLURM_ARRAY_TASK_ID]}}"

# ---- Configuration ----------------------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"
# HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
# HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
# HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.1-8B}"
# HF_MODEL="${HF_MODEL:-hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4}"
# HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"

AUTO_SUBMIT_RLHF=1  # set to 1 to auto-submit pipeline_rlhf.sh after finetune stage

# ---- Model-family detection -------------------------------------------------
# Automatically selects the correct vLLM venv, dtype, and chat template.
if [[ "$HF_MODEL" == *"Llama"* ]] || [[ "$HF_MODEL" == *"llama"* ]]; then
    IS_LLAMA=1
else
    IS_LLAMA=0
fi

# Thinking mode is Qwen3-only; disable unconditionally for Llama.
DISABLE_THINKING="${DISABLE_THINKING:-$([ "$IS_LLAMA" = "1" ] && echo 0 || echo 1)}"
RUN_NUM="${RUN_NUM:-}"   # run number; appends /{RUN_NUM} to MODEL_SLUG and offsets VLLM_PORT
# If MODEL_SLUG was passed in (e.g. for a custom experiment name), use it directly.
# Otherwise derive it from HF_MODEL and DISABLE_THINKING.
if [[ -z "${MODEL_SLUG:-}" ]]; then
    MODEL_SLUG="${HF_MODEL//\//-}"
    # Qwen3 slug includes thinking mode; other models omit it.
    if [[ "$IS_LLAMA" = "0" ]]; then
        MODEL_SLUG="${MODEL_SLUG}-exp3_fix_hang-$([ "${DISABLE_THINKING}" = "1" ] && echo no_thinking || echo thinking)"
    else
        MODEL_SLUG="${MODEL_SLUG}-exp3_fix_hang"
    fi
    [[ -n "${RUN_NUM}" ]] && MODEL_SLUG="${MODEL_SLUG}/${RUN_NUM}"
fi
export DISABLE_THINKING MODEL_SLUG PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

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

COLD_START_ROOT="${COLD_START_ROOT:-ColdStartData}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
GRAPHS_ROOT="${GRAPHS_ROOT:-graphs}"
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_results}"

COLD_START_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${COLD_START_ROOT}/${DATASET}"
CHECKPOINT_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${CHECKPOINT_ROOT}/${DATASET}"

# ---- vLLM inference server --------------------------------------------------
# Set USE_VLLM_SERVER=0 to disable and fall back to HuggingFace transformers.
USE_VLLM_SERVER="${USE_VLLM_SERVER:-1}"
VLLM_PORT="${VLLM_PORT:-$((7789 + 12 + ${SLURM_ARRAY_TASK_ID:-0} + ${RUN_NUM:-0} * 100))}"
VLLM_TP="${VLLM_TP:-1}"                  # tensor-parallel GPUs for the server
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
# Llama (GPTQ) uses llama_serve venv + dtype=auto + built-in chat template.
# Qwen3 uses vllm_serve venv + dtype=float16 + nonthinking Jinja override.
if [[ "$IS_LLAMA" = "1" ]]; then
    VLLM_SERVE_DIR="${VLLM_SERVE_DIR:-/home/users/psuwannapichat/work_space/llama_serve}"
    # VLLM_DTYPE="${VLLM_DTYPE:-auto}"
    VLLM_DTYPE="${VLLM_DTYPE:-float16}"
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
        --gpu-memory-utilization 0.90
        # --max-num-seqs           8
        --tensor-parallel-size   "$VLLM_TP"
        --enforce-eager)
    # Only pass --chat-template when a custom template is specified (Qwen3).
    # Llama Instruct models use their built-in template.
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

# _ensure_vllm_running — start vLLM if not already alive.
_ensure_vllm_running() {
    if [[ "$USE_VLLM_SERVER" != "1" ]]; then return 0; fi
    if [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "  vLLM server already running (PID ${VLLM_PID})."
        return 0
    fi
    _start_vllm_server
}

# _ensure_vllm_stopped — kill vLLM and wait for the process to exit so GPU
# VRAM is reclaimed before NN training / graph-generation stages start.
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
echo "  vLLM server      : $([ "$USE_VLLM_SERVER" = "1" ] && echo "enabled (port $VLLM_PORT, tp=$VLLM_TP)" || echo "disabled (HF backend)")"
echo "  Started at       : $(date)"
echo "════════════════════════════════════════════════"

if [[ "$USE_VLLM_SERVER" == "1" ]]; then
    export LOCAL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export LOCAL_API_KEY="EMPTY"
    export USE_VLLM_SERVER USE_VLLM=0
    # vLLM is started/stopped per-stage by _ensure_vllm_running/_ensure_vllm_stopped
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
    echo "  ✓ Pre-train → $CHECKPOINT_DIR"
fi

# ---- Stage 3a: Build D_eff (LLM inference — vLLM must be running) -----------
echo ""
if stage_done stage3a_build_deff || stage_done stage3_finetune; then
    echo "▶ [Stage 3a/4] Build D_eff ... SKIPPED (already done)"
else
    echo "▶ [Stage 3a/4] Build D_eff (LLM inference for D_pruned + D_simple) ..."
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

# ---- Stage 3b: NN fine-tune (no vLLM — GPU free for CUDA training) ----------
echo ""
if stage_done stage3b_nn_finetune || stage_done stage3_finetune; then
    echo "▶ [Stage 3b/4] NN fine-tune ... SKIPPED (already done)"
else
    echo "▶ [Stage 3b/4] NN fine-tune (ARGDesigner on D_eff, vLLM stopped) ..."
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

    # Auto-submit RLHF pipeline for this dataset as soon as finetune is done.
    # Only submitted once (guard prevents re-submission if stage is already marked done).
    if [[ "${AUTO_SUBMIT_RLHF:-1}" == "1" ]]; then
        RLHF_JOB=$(sbatch --parsable \
            --array="$SLURM_ARRAY_TASK_ID" \
            --export=ALL,\
MODEL_SLUG="$MODEL_SLUG",\
HF_MODEL="$HF_MODEL",\
DISABLE_THINKING="$DISABLE_THINKING",\
SEED="$SEED",\
CHECKPOINT_ROOT="$CHECKPOINT_ROOT",\
COLD_START_ROOT="$COLD_START_ROOT",\
GRAPHS_ROOT="$GRAPHS_ROOT",\
RESULTS_ROOT="$RESULTS_ROOT",\
EVAL_BATCH="$EVAL_BATCH",\
PRUNING_RATIO="$PRUNING_RATIO",\
VLLM_SERVE_DIR="$VLLM_SERVE_DIR",\
VLLM_DTYPE="$VLLM_DTYPE",\
VLLM_CHAT_TEMPLATE="$VLLM_CHAT_TEMPLATE",\
VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN",\
USE_VLLM_SERVER="$USE_VLLM_SERVER",\
VLLM_TP="$VLLM_TP",\
RUN_NUM="${RUN_NUM}",\
DIFFICULTY_FILTER="${DIFFICULTY_FILTER:-0}",\
MIN_FAIL_RATE="${MIN_FAIL_RATE:-0.05}",\
WEAK_BASELINES="${WEAK_BASELINES:-0}",\
AUTO_SUBMIT_RLHF=0 \
            "${SLURM_SUBMIT_DIR}/slurm/pipeline_rlhf.sh")
        echo "  ↳ Submitted RLHF pipeline for array task $SLURM_ARRAY_TASK_ID → job $RLHF_JOB"
        echo "     MODEL_SLUG=$MODEL_SLUG"
    fi
fi

# ---- Shared setup for stages 4a/4b ------------------------------------------
# Task-split paths (standard ordering: 0=gsm8k 1=aqua 2=multiarith 3=svamp 4=humaneval 5=mmlu)
TASK_SPLIT_PATHS=(
    "benchmark_datasets/gsm8k/task_split_gsm8k.json"
    "benchmark_datasets/AQuA/task_split_aqua.json"
    "benchmark_datasets/MultiArith/task_split_multiarith.json"
    "benchmark_datasets/SVAMP/task_split_svamp.json"
    "benchmark_datasets/humaneval/task_split_humaneval.json"
    "benchmark_datasets/MMLU/task_split_mmlu.json"
)
TASK_SPLIT_RAW="${TASK_SPLIT_PATHS[$SLURM_ARRAY_TASK_ID]}"

MODEL_TYPE="arg_designer"
GRAPHS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${GRAPHS_ROOT}/${MODEL_TYPE}"
GRAPHS_FILE="$GRAPHS_DIR/${DATASET}_graphs.jsonl"

DECISION_METHODS=(FinalRefer FinalRefer FinalRefer FinalRefer FinalWriteCode FinalRefer)
DECISION="${DECISION_METHODS[$SLURM_ARRAY_TASK_ID]}"

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
    _ensure_vllm_running

    RESULTS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/${MODEL_TYPE}"
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
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
        --model_type       "$MODEL_TYPE" \
        ${LIMIT:+--limit "$LIMIT"}

    mark_done stage4b_benchmark
    echo "  ✓ Results → $OUTPUT_FILE"
fi

# ---- Stage 5a: Generate graphs on TRAINING set (overfitting check) ----------
# Uses base_tasks_indices + finetune_tasks_indices from the task split so we
# can directly compare train vs test accuracy to diagnose overfitting.
echo ""
TRAIN_GRAPHS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${GRAPHS_ROOT}/${MODEL_TYPE}_train"
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
        --model_type   "${MODEL_TYPE}_train" \
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

    TRAIN_RESULTS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/${MODEL_TYPE}_train"
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
        --model_type       "${MODEL_TYPE}_train"

    mark_done stage5b_benchmark_train
    echo "  ✓ Train results → $TRAIN_OUTPUT_FILE"
fi

echo ""
echo "════════════════════════════════════════════════"
echo "  Pipeline complete for $DATASET"
echo "  Test  results : ${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/${MODEL_TYPE}/summary.jsonl"
echo "  Train results : ${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/${MODEL_TYPE}_train/summary.jsonl"
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════"
