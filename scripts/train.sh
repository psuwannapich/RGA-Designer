#!/bin/bash
# ---------------------------------------------------------------------------
# ARGDesigner training pipeline — Stage 1 (cold-start) → Stage 2 (pre-train)
#                                 → Stage 3 (fine-tune) → Stage 4 (benchmark)
#
# Usage:
#   DATASET_IDX=0 bash scripts/train.sh          # gsm8k
#   DATASET=gsm8k bash scripts/train.sh          # by name
#   bash scripts/train.sh 0                      # positional arg (0=gsm8k)
#
# Dataset index mapping:
#   0=gsm8k  1=aqua  2=multiarith  3=svamp  4=humaneval  5=mmlu
#
# Key environment variables (all optional):
#   HF_MODEL           HuggingFace model ID            (default: Qwen/Qwen3-4B)
#   VLLM_SERVE_DIR     directory containing vLLM .venv (required if USE_VLLM_SERVER=1)
#   USE_VLLM_SERVER    1 = use vLLM HTTP server        (default: 1)
#   VLLM_PORT          port for vLLM server            (default: 8000 + DATASET_IDX)
#   DISABLE_THINKING   1 = Qwen3 no-thinking mode      (default: 1)
#   NUM_TASKS          cold-start tasks (0 = all)       (default: 0)
#   EPOCHS             pre-train epochs                 (default: 30)
#   FINETUNE_EPOCHS    Phase-2 fine-tune epochs         (default: 30)
#   EVAL_BATCH         benchmark batch size             (default: 16)
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Dataset index ----------------------------------------------------------
DATASET_IDX="${1:-${DATASET_IDX:-0}}"
if [[ -n "${DATASET:-}" ]]; then
    case "$DATASET" in
        gsm8k)     DATASET_IDX=0 ;;
        aqua)      DATASET_IDX=1 ;;
        multiarith) DATASET_IDX=2 ;;
        svamp)     DATASET_IDX=3 ;;
        humaneval) DATASET_IDX=4 ;;
        mmlu)      DATASET_IDX=5 ;;
        *) echo "ERROR: unknown DATASET='$DATASET'"; exit 1 ;;
    esac
fi

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

DATASET="${DATASETS[$DATASET_IDX]}"
DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$DATASET_IDX]}"
TASK_SPLIT_RAW="${TASK_SPLIT_PATHS[$DATASET_IDX]}"
MIN_AGENTS="${MIN_AGENTS:-${DATASET_MIN_AGENTS[$DATASET_IDX]}}"
MAX_AGENTS="${MAX_AGENTS:-${DATASET_MAX_AGENTS[$DATASET_IDX]}}"
DECISION="${DECISION_METHODS[$DATASET_IDX]}"

# ---- Model configuration ----------------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"

if [[ -z "${MODEL_SLUG:-}" ]]; then
    MODEL_SLUG="${HF_MODEL//\//-}"
    MODEL_SLUG="${MODEL_SLUG}-$([ "${DISABLE_THINKING}" = "1" ] && echo no_thinking || echo thinking)"
fi
export DISABLE_THINKING MODEL_SLUG PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

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

COLD_START_ROOT="${COLD_START_ROOT:-ColdStartData}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
GRAPHS_ROOT="${GRAPHS_ROOT:-graphs}"
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_results}"

COLD_START_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${COLD_START_ROOT}/${DATASET}"
CHECKPOINT_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${CHECKPOINT_ROOT}/${DATASET}"

# ---- vLLM server ------------------------------------------------------------
USE_VLLM_SERVER="${USE_VLLM_SERVER:-1}"
VLLM_PORT="${VLLM_PORT:-$((8000 + DATASET_IDX))}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"
# Point VLLM_SERVE_DIR to the directory containing your vLLM .venv.
# Required when USE_VLLM_SERVER=1.
VLLM_SERVE_DIR="${VLLM_SERVE_DIR:-}"
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-}"   # optional Jinja template override
VLLM_PID=""

_start_vllm_server() {
    if [[ -z "$VLLM_SERVE_DIR" ]]; then
        echo "ERROR: VLLM_SERVE_DIR is not set. Export it before running this script."
        exit 1
    fi
    echo "Starting vLLM server for '$HF_MODEL' on port $VLLM_PORT ..."
    mkdir -p "$PROJECT_ROOT/logs"
    local _cmd=("$VLLM_SERVE_DIR/.venv/bin/vllm" serve "$HF_MODEL"
        --port "$VLLM_PORT" --dtype "$VLLM_DTYPE" --trust-remote-code
        --max-model-len "$VLLM_MAX_MODEL_LEN"
        --gpu-memory-utilization 0.90
        --tensor-parallel-size "$VLLM_TP" --enforce-eager)
    [[ -n "${VLLM_CHAT_TEMPLATE:-}" ]] && _cmd+=(--chat-template "$VLLM_CHAT_TEMPLATE")
    ("${_cmd[@]}") > "$PROJECT_ROOT/logs/vllm_$$.log" 2>&1 &
    VLLM_PID=$!
    echo "  PID=$VLLM_PID  log=logs/vllm_$$.log"
    for _i in $(seq 1 120); do
        curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 && \
            echo "  vLLM ready (${_i}x5s)" && return 0
        sleep 5
    done
    echo "ERROR: vLLM did not start within 10 minutes." >&2; exit 1
}

_stop_vllm_server() {
    [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null || true
}
trap _stop_vllm_server EXIT

_ensure_vllm_running() {
    [[ "$USE_VLLM_SERVER" != "1" ]] && return 0
    [[ -n "${VLLM_PID:-}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null && return 0
    _start_vllm_server
}

_ensure_vllm_stopped() {
    [[ "$USE_VLLM_SERVER" != "1" ]] && return 0
    [[ -z "${VLLM_PID:-}" ]] || ! kill -0 "${VLLM_PID}" 2>/dev/null && return 0
    echo "Stopping vLLM server (PID ${VLLM_PID}) to free GPU VRAM ..."
    kill "${VLLM_PID}" 2>/dev/null || true
    local _w=0
    while kill -0 "${VLLM_PID}" 2>/dev/null; do
        sleep 2; _w=$((_w + 2))
        ((_w >= 60)) && { kill -9 "${VLLM_PID}" 2>/dev/null || true; break; }
    done
    VLLM_PID=""
    sleep 5
}

# ---- Stage checkpointing ----------------------------------------------------
STATE_DIR="$PROJECT_ROOT/${MODEL_SLUG}/state/${DATASET}"
mkdir -p "$STATE_DIR"
stage_done() { [[ -f "$STATE_DIR/$1.done" ]]; }
mark_done()  { touch "$STATE_DIR/$1.done"; echo "  checkpoint: $STATE_DIR/$1.done"; }

echo "================================================"
echo "  ARGDesigner Training Pipeline"
echo "  Dataset  : $DATASET (index $DATASET_IDX)"
echo "  Model    : $HF_MODEL  (slug: $MODEL_SLUG)"
echo "  vLLM     : $([ "$USE_VLLM_SERVER" = "1" ] && echo "port $VLLM_PORT, tp=$VLLM_TP" || echo "disabled (HF backend)")"
echo "  Started  : $(date)"
echo "================================================"

if [[ "$USE_VLLM_SERVER" == "1" ]]; then
    export LOCAL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export LOCAL_API_KEY="EMPTY"
    export USE_VLLM_SERVER USE_VLLM=0
fi

# ---- Stage 1: Cold-start ----------------------------------------------------
if stage_done stage1_cold_start; then
    echo "[Stage 1] Cold-start ... SKIPPED"
else
    echo "[Stage 1] Cold-start ..."
    _ensure_vllm_running

    if [[ "$DATASET" == "mmlu" && ! -d "$DATASET_JSON/test" ]]; then
        uv run python benchmark_datasets/MMLU/download.py
    fi

    uv run cold-start \
        --dataset "$DATASET" --dataset_json "$DATASET_JSON" \
        --llm_name "$HF_MODEL" --output_dir "$COLD_START_DIR" \
        --num_tasks "$NUM_TASKS" --num_iterations "$NUM_ITERATIONS" \
        --batch_size "$BATCH_SIZE" --num_rounds "$NUM_ROUNDS" \
        --min_agents "$MIN_AGENTS" --max_agents "$MAX_AGENTS" --seed "$SEED"

    mark_done stage1_cold_start
fi

# ---- Stage 2: Pre-train -----------------------------------------------------
if stage_done stage2_pretrain; then
    echo "[Stage 2] Pre-train ... SKIPPED"
else
    echo "[Stage 2] Pre-train ..."
    _ensure_vllm_stopped

    uv run python experiment/pretrain.py \
        --dataset "$DATASET" --data_dir "$COLD_START_DIR" \
        --output_dir "$CHECKPOINT_DIR" --epochs "$EPOCHS" \
        --lr "$TRAIN_LR" --batch_size "$TRAIN_BATCH_SIZE" \
        --val_ratio "$VAL_RATIO" --seed "$SEED"

    mark_done stage2_pretrain
fi

# ---- Stage 3a: Build D_eff --------------------------------------------------
if stage_done stage3a_build_deff || stage_done stage3_finetune; then
    echo "[Stage 3a] Build D_eff ... SKIPPED"
else
    echo "[Stage 3a] Build D_eff ..."
    _ensure_vllm_running

    uv run python experiment/finetune_gemma.py \
        --phase build_deff --dataset "$DATASET" --dataset_json "$DATASET_JSON" \
        --cold_start_dir "$COLD_START_DIR" --checkpoint_dir "$CHECKPOINT_DIR" \
        --output_dir "$CHECKPOINT_DIR" --llm_name "$HF_MODEL" \
        --pruning_ratio "$PRUNING_RATIO" --replay_ratio "$REPLAY_RATIO" \
        --batch_size "$BATCH_SIZE" --seed "$SEED"

    mark_done stage3a_build_deff
fi

# ---- Stage 3b: NN fine-tune -------------------------------------------------
if stage_done stage3b_nn_finetune || stage_done stage3_finetune; then
    echo "[Stage 3b] NN fine-tune ... SKIPPED"
else
    echo "[Stage 3b] NN fine-tune ..."
    _ensure_vllm_stopped

    uv run python experiment/finetune_gemma.py \
        --phase finetune --dataset "$DATASET" --dataset_json "$DATASET_JSON" \
        --cold_start_dir "$COLD_START_DIR" --checkpoint_dir "$CHECKPOINT_DIR" \
        --output_dir "$CHECKPOINT_DIR" --llm_name "$HF_MODEL" \
        --finetune_epochs "$FINETUNE_EPOCHS" --finetune_lr "$FINETUNE_LR" \
        --train_batch_size "$TRAIN_BATCH_SIZE" --seed "$SEED"

    mark_done stage3b_nn_finetune
fi

MODEL_TYPE="arg_designer"
GRAPHS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${GRAPHS_ROOT}/${MODEL_TYPE}"
GRAPHS_FILE="$GRAPHS_DIR/${DATASET}_graphs.jsonl"
NO_EF_FLAG=""; [[ "$NO_EF" == "1" ]] && NO_EF_FLAG="--no_ef"

# ---- Stage 4a: Generate graphs ----------------------------------------------
if stage_done stage4a_gen_graphs; then
    echo "[Stage 4a] Generate graphs ... SKIPPED"
else
    echo "[Stage 4a] Generate graphs ..."
    mkdir -p "$GRAPHS_DIR"

    uv run python experiment/generate_graphs.py \
        --model_path "$CHECKPOINT_DIR" --dataset "$DATASET" \
        --dataset_path "$DATASET_JSON" --output_file "$GRAPHS_FILE" \
        --model_type "$MODEL_TYPE" \
        ${TASK_SPLIT_RAW:+--task_split_path "$PROJECT_ROOT/$TASK_SPLIT_RAW"} \
        ${LIMIT:+--limit "$LIMIT"} $NO_EF_FLAG

    mark_done stage4a_gen_graphs
fi

# ---- Stage 4b: Benchmark ----------------------------------------------------
if stage_done stage4b_benchmark; then
    echo "[Stage 4b] Benchmark ... SKIPPED"
else
    echo "[Stage 4b] Benchmark ..."
    _ensure_vllm_running

    RESULTS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/${MODEL_TYPE}"
    mkdir -p "$RESULTS_DIR"

    uv run python experiment/benchmark_pregraph.py \
        --graphs_file "$GRAPHS_FILE" --dataset "$DATASET" \
        --llm_name "$HF_MODEL" --decision_method "$DECISION" \
        --output_file "$RESULTS_DIR/${DATASET}.jsonl" \
        --summary_log_file "$RESULTS_DIR/summary.jsonl" \
        --eval_batch_size "$EVAL_BATCH" --model_type "$MODEL_TYPE" \
        ${LIMIT:+--limit "$LIMIT"}

    mark_done stage4b_benchmark
fi

echo ""
echo "================================================"
echo "  Training pipeline complete: $DATASET"
echo "  Results: ${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/${MODEL_TYPE}/summary.jsonl"
echo "  Finished: $(date)"
echo "================================================"
