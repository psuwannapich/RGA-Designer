#!/bin/bash
# ---------------------------------------------------------------------------
# RLHF pipeline — collect → train_rm → train_policy → benchmark
#
# Prerequisites: scripts/train.sh must have completed so that
#   ${MODEL_SLUG}/checkpoints/${DATASET}/ef_best_model.pth exists.
#
# Usage:
#   DATASET_IDX=0 bash scripts/rlhf.sh          # gsm8k
#   DATASET=gsm8k bash scripts/rlhf.sh          # by name
#   bash scripts/rlhf.sh 0                      # positional arg
#
# Dataset index mapping:
#   0=gsm8k  1=aqua  2=multiarith  3=svamp  4=humaneval  5=mmlu
#
# Key environment variables (all optional):
#   HF_MODEL           HuggingFace model ID            (default: Qwen/Qwen3-4B)
#   VLLM_SERVE_DIR     directory containing vLLM .venv (required if USE_VLLM_SERVER=1)
#   USE_VLLM_SERVER    1 = use vLLM HTTP server        (default: 1)
#   VLLM_PORT          port for vLLM server            (default: 8100 + DATASET_IDX)
#   DISABLE_THINKING   1 = Qwen3 no-thinking mode      (default: 1)
#   RLHF_NUM_TASKS     tasks for preference collection  (default: 100)
#   KL_COEFF           KL penalty for policy training   (default: 0.2)
#   BEST_OF_N          BoN candidates at inference      (default: 5)
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Dataset index ----------------------------------------------------------
DATASET_IDX="${1:-${DATASET_IDX:-0}}"
if [[ -n "${DATASET:-}" ]]; then
    case "$DATASET" in
        gsm8k)      DATASET_IDX=0 ;;
        aqua)       DATASET_IDX=1 ;;
        multiarith) DATASET_IDX=2 ;;
        svamp)      DATASET_IDX=3 ;;
        humaneval)  DATASET_IDX=4 ;;
        mmlu)       DATASET_IDX=5 ;;
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
DATASET_MAX_AGENTS=(4 4 4 4 5 6)
DECISION_METHODS=(FinalRefer FinalRefer FinalRefer FinalRefer FinalWriteCode FinalRefer)

DATASET="${DATASETS[$DATASET_IDX]}"
DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$DATASET_IDX]}"
TASK_SPLIT_RAW="${TASK_SPLIT_PATHS[$DATASET_IDX]}"
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
SEED="${SEED:-42}"

# Collect
RLHF_NUM_TASKS="${RLHF_NUM_TASKS:-100}"
PREFERENCE_ROOT="${PREFERENCE_ROOT:-rlhf_data}"
MIN_AGENTS="${MIN_AGENTS:-2}"
W_CORRECT="${W_CORRECT:-0.6}"
W_SIZE="${W_SIZE:-0.3}"
W_EDGE="${W_EDGE:-0.1}"
PAIR_MARGIN="${PAIR_MARGIN:-0.1}"
PRUNING_RATIO="${PRUNING_RATIO:-0.25}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2}"
TASK_CONCURRENCY="${TASK_CONCURRENCY:-8}"
INFERENCE_CONCURRENCY="${INFERENCE_CONCURRENCY:-8}"
LLM_TIMEOUT="${LLM_TIMEOUT:-2400}"
SAMPLE_TEMPERATURES="${SAMPLE_TEMPERATURES:-0.5 1.0 1.5 2.0}"
ARG_MODEL_DIR="${ARG_MODEL_DIR:-}"
COLDSTART_DIRS="${COLDSTART_DIRS:-}"
ARG_MODEL_SAMPLES="${ARG_MODEL_SAMPLES:-}"
COLD_START_ROOT="${COLD_START_ROOT:-ColdStartData}"
DIFFICULTY_FILTER="${DIFFICULTY_FILTER:-0}"
MIN_FAIL_RATE="${MIN_FAIL_RATE:-0.05}"
WEAK_BASELINES="${WEAK_BASELINES:-1}"
ROLE_SWEEP="${ROLE_SWEEP:-1}"
ROLE_SWEEP_TOPOLOGY="${ROLE_SWEEP_TOPOLOGY:-Chain}"
ROLE_SWEEP_N_AGENTS="${ROLE_SWEEP_N_AGENTS:-2}"
ROLE_SWEEP_MAX_COMBOS="${ROLE_SWEEP_MAX_COMBOS:-}"
BOTH_WRONG_WEIGHT="${BOTH_WRONG_WEIGHT:-0}"
BOTH_CORRECT_WEIGHT="${BOTH_CORRECT_WEIGHT:-0}"

# Reward model
RM_ROOT="${RM_ROOT:-rlhf_checkpoints}"
RM_EPOCHS="${RM_EPOCHS:-30}"
RM_LR="${RM_LR:-1e-4}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-32}"
RM_HIDDEN_DIM="${RM_HIDDEN_DIM:-256}"
RM_OUTPUT_DIM="${RM_OUTPUT_DIM:-128}"
RM_VAL_FRACTION="${RM_VAL_FRACTION:-0.1}"

# Policy
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
POLICY_ROOT="${POLICY_ROOT:-rlhf_checkpoints}"
POLICY_EPOCHS="${POLICY_EPOCHS:-30}"
POLICY_NUM_TASKS="${POLICY_NUM_TASKS:-100}"
POLICY_LR="${POLICY_LR:-5e-6}"
KL_COEFF="${KL_COEFF:-0.2}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"

# Benchmark
EVAL_BATCH="${EVAL_BATCH:-8}"
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_results}"
GRAPHS_ROOT="${GRAPHS_ROOT:-graphs}"
BEST_OF_N="${BEST_OF_N:-5}"
BON_TEMPERATURE="${BON_TEMPERATURE:-1}"
LIMIT="${LIMIT:-}"
NO_EF="${NO_EF:-0}"

# Derived paths
PREFERENCE_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${PREFERENCE_ROOT}/${DATASET}"
RM_CHECKPOINT="$PROJECT_ROOT/${MODEL_SLUG}/${RM_ROOT}/${DATASET}/reward_model.pth"
POLICY_CHECKPOINT="$PROJECT_ROOT/${MODEL_SLUG}/${POLICY_ROOT}/${DATASET}/policy_rlhf.pth"
MODEL_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${CHECKPOINT_ROOT}/${DATASET}"
COLD_START_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${COLD_START_ROOT}/${DATASET}"
D_EFF_DIR="$MODEL_DIR/FinetuneData_${DATASET}"
ARG_MODEL_DIR="${ARG_MODEL_DIR:-$MODEL_DIR}"

# ---- vLLM server ------------------------------------------------------------
USE_VLLM_SERVER="${USE_VLLM_SERVER:-1}"
VLLM_PORT="${VLLM_PORT:-$((8100 + DATASET_IDX))}"
VLLM_TP="${VLLM_TP:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_DTYPE="${VLLM_DTYPE:-float16}"
VLLM_SERVE_DIR="${VLLM_SERVE_DIR:-}"
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-}"
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
    echo "Stopping vLLM server (PID ${VLLM_PID}) ..."
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
STATE_DIR="$PROJECT_ROOT/${MODEL_SLUG}/state/rlhf/${DATASET}"
mkdir -p "$STATE_DIR"
stage_done() { [[ -f "$STATE_DIR/$1.done" ]]; }
mark_done()  { touch "$STATE_DIR/$1.done"; echo "  checkpoint: $STATE_DIR/$1.done"; }

echo "================================================"
echo "  RLHF Pipeline"
echo "  Dataset  : $DATASET (index $DATASET_IDX)"
echo "  Model    : $HF_MODEL  (slug: $MODEL_SLUG)"
echo "  vLLM     : $([ "$USE_VLLM_SERVER" = "1" ] && echo "port $VLLM_PORT, tp=$VLLM_TP" || echo "disabled (HF backend)")"
echo "  BoN      : $([ "${BEST_OF_N:-1}" -gt 1 ] && echo "N=$BEST_OF_N temp=$BON_TEMPERATURE" || echo "disabled")"
echo "  Started  : $(date)"
echo "================================================"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: base checkpoint not found: $MODEL_DIR"
    echo "  Run scripts/train.sh first."
    exit 1
fi

if [[ "$USE_VLLM_SERVER" == "1" ]]; then
    export LOCAL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export LOCAL_API_KEY="EMPTY"
    export USE_VLLM_SERVER USE_VLLM=0
fi

# Helper to build --coldstart_dirs flag
_coldstart_dirs_flag() {
    local _dirs="${COLDSTART_DIRS:-}"
    [[ -d "$COLD_START_DIR" ]] && _dirs="${_dirs:+$_dirs }$COLD_START_DIR"
    [[ -d "$D_EFF_DIR"      ]] && _dirs="${_dirs:+$_dirs }$D_EFF_DIR"
    [[ -d "$COLD_START_DIR/rlhf_rejected" ]] && _dirs="${_dirs:+$_dirs }$COLD_START_DIR/rlhf_rejected"
    [[ -d "$D_EFF_DIR/rlhf_rejected"      ]] && _dirs="${_dirs:+$_dirs }$D_EFF_DIR/rlhf_rejected"
    [[ -n "$_dirs" ]] && echo "--coldstart_dirs $_dirs"
}

# ---- Stage 1a: Generate candidates ------------------------------------------
if stage_done stage1a_gen_candidates || stage_done stage1_collect; then
    echo "[Stage 1a] Gen candidates ... SKIPPED"
else
    echo "[Stage 1a] Gen candidates ..."
    _ensure_vllm_stopped

    uv run rlhf \
        --dataset "$DATASET" --phase gen_candidates \
        --llm_name "$HF_MODEL" --dataset_json "$DATASET_JSON" \
        --num_tasks "$RLHF_NUM_TASKS" --preference_dir "$PREFERENCE_DIR" \
        --min_agents "$MIN_AGENTS" --max_agents "$MAX_AGENTS" \
        --w_correct "$W_CORRECT" --w_size "$W_SIZE" --w_edge "$W_EDGE" \
        --pair_margin "$PAIR_MARGIN" --seed "$SEED" \
        --sample_temperatures $SAMPLE_TEMPERATURES \
        ${ARG_MODEL_DIR:+--arg_model_dir "$ARG_MODEL_DIR"} \
        ${ARG_MODEL_SAMPLES:+--arg_model_samples "$ARG_MODEL_SAMPLES"} \
        $([ "${DIFFICULTY_FILTER}" = "1" ] && echo "--difficulty_filter --min_fail_rate $MIN_FAIL_RATE") \
        $([ "${WEAK_BASELINES}"    = "1" ] && echo "--weak_baselines") \
        $([ "${ROLE_SWEEP}"        = "1" ] && echo "--role_sweep --role_sweep_topology $ROLE_SWEEP_TOPOLOGY --role_sweep_n_agents $ROLE_SWEEP_N_AGENTS") \
        ${ROLE_SWEEP_MAX_COMBOS:+$([ "${ROLE_SWEEP}" = "1" ] && echo "--role_sweep_max_combos $ROLE_SWEEP_MAX_COMBOS")} \
        $(_coldstart_dirs_flag)

    mark_done stage1a_gen_candidates
fi

# ---- Stage 1b: LLM scoring --------------------------------------------------
if stage_done stage1b_collect_llm || stage_done stage1_collect; then
    echo "[Stage 1b] Collect LLM scoring ... SKIPPED"
else
    echo "[Stage 1b] Collect LLM scoring ..."
    _ensure_vllm_running

    uv run rlhf \
        --dataset "$DATASET" --phase collect_llm \
        --llm_name "$HF_MODEL" --dataset_json "$DATASET_JSON" \
        --num_tasks "$RLHF_NUM_TASKS" --preference_dir "$PREFERENCE_DIR" \
        --min_agents "$MIN_AGENTS" --max_agents "$MAX_AGENTS" \
        --w_correct "$W_CORRECT" --w_size "$W_SIZE" --w_edge "$W_EDGE" \
        --pair_margin "$PAIR_MARGIN" --pruning_ratio "$PRUNING_RATIO" \
        --checkpoint_every "$CHECKPOINT_EVERY" \
        --task_concurrency "$TASK_CONCURRENCY" \
        --inference_concurrency "$INFERENCE_CONCURRENCY" \
        --llm_timeout "$LLM_TIMEOUT" --seed "$SEED" \
        $(_coldstart_dirs_flag)

    mark_done stage1b_collect_llm
fi

# ---- Stage 2: Train reward model --------------------------------------------
if stage_done stage2_train_rm; then
    echo "[Stage 2] Train reward model ... SKIPPED"
else
    echo "[Stage 2] Train reward model ..."
    _ensure_vllm_stopped

    uv run rlhf \
        --dataset "$DATASET" --phase train_rm \
        --preference_dir "$PREFERENCE_DIR" \
        --rm_checkpoint "$RM_CHECKPOINT" \
        --rm_epochs "$RM_EPOCHS" --rm_lr "$RM_LR" \
        --rm_batch_size "$RM_BATCH_SIZE" \
        --rm_hidden_dim "$RM_HIDDEN_DIM" --rm_output_dim "$RM_OUTPUT_DIM" \
        --rm_val_fraction "$RM_VAL_FRACTION" \
        --both_wrong_weight "$BOTH_WRONG_WEIGHT" \
        --both_correct_weight "$BOTH_CORRECT_WEIGHT"

    mark_done stage2_train_rm
fi

# ---- Stage 3: Train policy --------------------------------------------------
if stage_done stage3_train_policy; then
    echo "[Stage 3] Train policy ... SKIPPED"
else
    echo "[Stage 3] Train policy ..."

    uv run rlhf \
        --dataset "$DATASET" --phase train_policy \
        --llm_name "$HF_MODEL" --model_dir "$MODEL_DIR" \
        --rm_checkpoint "$RM_CHECKPOINT" \
        --policy_checkpoint "$POLICY_CHECKPOINT" \
        --dataset_json "$DATASET_JSON" \
        --num_tasks "$POLICY_NUM_TASKS" \
        --policy_epochs "$POLICY_EPOCHS" --policy_lr "$POLICY_LR" \
        --kl_coeff "$KL_COEFF" --samples_per_task "$SAMPLES_PER_TASK" \
        --grad_accum_steps "$GRAD_ACCUM_STEPS" --seed "$SEED"

    mark_done stage3_train_policy
fi

POLICY_DIR="$(dirname "$POLICY_CHECKPOINT")"
MODEL_TYPE="rlhf"
GRAPHS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${GRAPHS_ROOT}/${MODEL_TYPE}"
GRAPHS_FILE="$GRAPHS_DIR/${DATASET}_graphs.jsonl"
NO_EF_FLAG=""; [[ "$NO_EF" == "1" ]] && NO_EF_FLAG="--no_ef"

# Best-of-N flags
BON_FLAGS=""
if [[ "${BEST_OF_N:-1}" -gt 1 ]]; then
    ROLE_EMB_FILE="$COLD_START_DIR/precomputed_role_embeddings.pkl"
    if [[ -f "$RM_CHECKPOINT" && -f "$ROLE_EMB_FILE" ]]; then
        BON_FLAGS="--best_of_n $BEST_OF_N --bon_temperature $BON_TEMPERATURE --rm_checkpoint $RM_CHECKPOINT --role_emb_path $ROLE_EMB_FILE"
        echo "  [BoN] N=$BEST_OF_N  temperature=$BON_TEMPERATURE"
    else
        echo "  [BoN] WARNING: RM or role embeddings missing — BoN disabled"
    fi
fi

# ---- Stage 4a: Generate graphs ----------------------------------------------
if stage_done stage4a_gen_graphs; then
    echo "[Stage 4a] Generate graphs ... SKIPPED"
else
    echo "[Stage 4a] Generate graphs ..."
    mkdir -p "$GRAPHS_DIR"

    uv run python experiment/generate_graphs.py \
        --model_path "$POLICY_DIR" --dataset "$DATASET" \
        --dataset_path "$DATASET_JSON" --output_file "$GRAPHS_FILE" \
        --model_type "$MODEL_TYPE" \
        ${TASK_SPLIT_RAW:+--task_split_path "$PROJECT_ROOT/$TASK_SPLIT_RAW"} \
        ${LIMIT:+--limit "$LIMIT"} $NO_EF_FLAG $BON_FLAGS

    mark_done stage4a_gen_graphs
fi

# ---- Stage 4b: Benchmark ----------------------------------------------------
if stage_done stage4b_benchmark; then
    echo "[Stage 4b] Benchmark ... SKIPPED"
else
    echo "[Stage 4b] Benchmark ..."
    _ensure_vllm_running

    RESULTS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/rlhf"
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
echo "  RLHF pipeline complete: $DATASET"
echo "  Results: ${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/rlhf/summary.jsonl"
echo "  Finished: $(date)"
echo "================================================"
