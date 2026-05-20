#!/bin/bash
#SBATCH --job-name=arg_qwen4_rlhf
#SBATCH --output=logs/qwen4_rlhf_%A_%a.out
#SBATCH --error=logs/qwen4_rlhf_%A_%a.err
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
# RLHF pipeline — all stages in one job per dataset.
#
# Stages (sequential within each array task):
#   1. collect       collect preference pairs via LLM graph runs
#   2. train_rm      train GNN reward model on preference pairs
#   3. train_policy  REINFORCE + KL fine-tuning of ARGDesigner policy
#   4. benchmark     evaluate the RLHF policy
#
# Prerequisites: pipeline_train.sh must have completed so that
#   ${MODEL_SLUG}/${CHECKPOINT_ROOT}/${DATASET}/ef_best_model.pth exists.
#
# Array index → dataset:
#   0 gsm8k  1 aqua  2 multiarith  3 svamp  4 humaneval  5 mmlu
#
# Usage:
#   sbatch slurm/pipeline_rlhf.sh
#   sbatch --array=0 slurm/pipeline_rlhf.sh       # gsm8k only
#
# Optional env vars:
#   HF_MODEL           HuggingFace model ID            (default: Qwen/Qwen3-8B)
#   DISABLE_THINKING   1 = no-thinking mode (Qwen3)    (default: 1)
#   RUN_NUM            run number — appended to MODEL_SLUG and offsets VLLM_PORT
#                      by RUN_NUM*100 to avoid collisions (default: empty)
#   CHECKPOINT_ROOT    Phase-1/2 checkpoint sub-dir     (default: checkpoints)
#   RLHF_NUM_TASKS     tasks to sample for collect      (default: 100)
#   PREFERENCE_ROOT    preference data sub-dir           (default: rlhf_data)
#   RM_ROOT            reward model checkpoint sub-dir   (default: rlhf_checkpoints)
#   POLICY_ROOT        policy checkpoint sub-dir         (default: rlhf_checkpoints)
#   COLDSTART_DIRS     space-separated .pt dirs for free pairs (optional)
#   ARG_MODEL_SAMPLES  unique ARGDesigner graphs per task (optional)
#   EVAL_BATCH         benchmark batch size              (default: 8)
#   RESULTS_ROOT       results sub-dir                  (default: benchmark_results)
#   BEST_OF_N          Best-of-N candidates at graph gen (default: 1 = disabled)
#   BON_TEMPERATURE    sampling temperature for BoN      (default: 1.2)
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
DATASET_MAX_AGENTS=(4 4 4 4 5 6)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$SLURM_ARRAY_TASK_ID]}"
MAX_AGENTS="${MAX_AGENTS:-${DATASET_MAX_AGENTS[$SLURM_ARRAY_TASK_ID]}}"

# ---- Configuration ----------------------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"
# HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
# HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
# HF_MODEL="${HF_MODEL:-meta-llama/Llama-3.1-8B}"

# ---- Model-family detection -------------------------------------------------
# Automatically selects the correct vLLM venv, dtype, and chat template.
# Add more patterns here if supporting additional model families.
if [[ "$HF_MODEL" == *"Llama"* ]] || [[ "$HF_MODEL" == *"llama"* ]]; then
    IS_LLAMA=1
else
    IS_LLAMA=0
fi

# Thinking mode is Qwen3-only; disable unconditionally for Llama.
DISABLE_THINKING="${DISABLE_THINKING:-$([ "$IS_LLAMA" = "1" ] && echo 0 || echo 1)}"
RUN_NUM="${RUN_NUM:-}"   # run number; must match pipeline_train.sh when auto-submitted
# If MODEL_SLUG was passed in (e.g. auto-submitted from pipeline_train.sh), use it
# directly so both scripts share the same directory.  Otherwise derive it the same
# way as pipeline_train.sh so the directories are consistent.
if [[ -z "${MODEL_SLUG:-}" ]]; then
    MODEL_SLUG="${HF_MODEL//\//-}"
    # Qwen3 slug includes thinking mode; other models omit it.
    if [[ "$IS_LLAMA" = "0" ]]; then
        MODEL_SLUG="${MODEL_SLUG}-$([ "${DISABLE_THINKING}" = "1" ] && echo no_thinking || echo thinking)"
    fi
    [[ -n "${RUN_NUM}" ]] && MODEL_SLUG="${MODEL_SLUG}/${RUN_NUM}"
fi
export DISABLE_THINKING MODEL_SLUG PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

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
ARG_MODEL_DIR="${ARG_MODEL_DIR:-}"         # defaults to MODEL_DIR after derived paths
COLDSTART_DIRS="${COLDSTART_DIRS:-}"       # user-specified extra dirs
ARG_MODEL_SAMPLES="${ARG_MODEL_SAMPLES:-}"
COLD_START_ROOT="${COLD_START_ROOT:-ColdStartData}"
DIFFICULTY_FILTER="${DIFFICULTY_FILTER:-0}"  # 1 = drop all-correct tasks before collecting
MIN_FAIL_RATE="${MIN_FAIL_RATE:-0.05}"        # threshold for difficulty filter
WEAK_BASELINES="${WEAK_BASELINES:-1}"         # 1 = add 1-agent and over-sized configs
ROLE_SWEEP="${ROLE_SWEEP:-1}"                 # 1 = enumerate all role combos on fixed topology
ROLE_SWEEP_TOPOLOGY="${ROLE_SWEEP_TOPOLOGY:-Chain}"
ROLE_SWEEP_N_AGENTS="${ROLE_SWEEP_N_AGENTS:-2}"      # comma-separated, e.g. "2,3"
ROLE_SWEEP_MAX_COMBOS="${ROLE_SWEEP_MAX_COMBOS:-}"   # empty = no cap

# Reward model
RM_ROOT="${RM_ROOT:-rlhf_checkpoints}"
RM_EPOCHS="${RM_EPOCHS:-30}"
RM_LR="${RM_LR:-1e-4}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-32}"
RM_HIDDEN_DIM="${RM_HIDDEN_DIM:-256}"
RM_OUTPUT_DIM="${RM_OUTPUT_DIM:-128}"
RM_VAL_FRACTION="${RM_VAL_FRACTION:-0.1}"
BOTH_WRONG_WEIGHT="${BOTH_WRONG_WEIGHT:-0}"
BOTH_CORRECT_WEIGHT="${BOTH_CORRECT_WEIGHT:-0}"

# Policy
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
POLICY_ROOT="${POLICY_ROOT:-rlhf_checkpoints}"
POLICY_EPOCHS="${POLICY_EPOCHS:-30}"
POLICY_NUM_TASKS="${POLICY_NUM_TASKS:-100}"
POLICY_LR="${POLICY_LR:-5e-6}"
KL_COEFF="${KL_COEFF:-0.1}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"

# Benchmark
EVAL_BATCH="${EVAL_BATCH:-8}"
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_results}"

# Best-of-N graph selection (inference-time scaling)
BEST_OF_N="${BEST_OF_N:-5}"           # 1 = disabled; >1 = generate N candidates, keep highest-RM-score
BON_TEMPERATURE="${BON_TEMPERATURE:-1}"  # diversity temperature for candidate sampling

# Derived paths
PREFERENCE_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${PREFERENCE_ROOT}/${DATASET}"
RM_CHECKPOINT="$PROJECT_ROOT/${MODEL_SLUG}/${RM_ROOT}/${DATASET}/reward_model.pth"
POLICY_CHECKPOINT="$PROJECT_ROOT/${MODEL_SLUG}/${POLICY_ROOT}/${DATASET}/policy_rlhf.pth"
MODEL_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${CHECKPOINT_ROOT}/${DATASET}"
COLD_START_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${COLD_START_ROOT}/${DATASET}"
D_EFF_DIR="$MODEL_DIR/FinetuneData_${DATASET}"
# Default ARG_MODEL_DIR to the fine-tuned checkpoint for this dataset
ARG_MODEL_DIR="${ARG_MODEL_DIR:-$MODEL_DIR}"

# ---- vLLM inference server --------------------------------------------------
# Set USE_VLLM_SERVER=0 to disable and fall back to HuggingFace transformers.
USE_VLLM_SERVER="${USE_VLLM_SERVER:-1}"
VLLM_PORT="${VLLM_PORT:-$((7789 + ${SLURM_ARRAY_TASK_ID:-0} + 21 + ${RUN_NUM:-0} * 100))}"
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

# _ensure_vllm_stopped — kill vLLM and wait for VRAM to be reclaimed before
# starting NN training stages (reward model, policy fine-tuning).
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
#   rm ${MODEL_SLUG}/state/rlhf/${DATASET}/stage1_collect.done
STATE_DIR="$PROJECT_ROOT/${MODEL_SLUG}/state/rlhf/${DATASET}"
mkdir -p "$STATE_DIR"

stage_done()  { [[ -f "$STATE_DIR/$1.done" ]]; }
mark_done()   { touch "$STATE_DIR/$1.done"; echo "  ✔ checkpoint written: $STATE_DIR/$1.done"; }

echo "════════════════════════════════════════════════"
echo "  RLHF Pipeline"
echo "  Job $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "  Node             : $SLURM_NODELIST"
echo "  Dataset          : $DATASET"
echo "  Model            : $HF_MODEL  (slug: $MODEL_SLUG)"
echo "  Collect tasks    : $RLHF_NUM_TASKS"
echo "  Policy tasks    : $POLICY_NUM_TASKS"
echo "  RM epochs        : $RM_EPOCHS"
echo "  Policy epochs    : $POLICY_EPOCHS"
echo "  Base model dir   : $MODEL_DIR"
echo "  vLLM server      : $([ "$USE_VLLM_SERVER" = "1" ] && echo "enabled (port $VLLM_PORT, tp=$VLLM_TP)" || echo "disabled (HF backend)")"
echo "  Best-of-N        : $([ "${BEST_OF_N:-1}" -gt 1 ] && echo "N=$BEST_OF_N  temp=$BON_TEMPERATURE" || echo "disabled")"
echo "  Started at       : $(date)"
echo "════════════════════════════════════════════════"

if [[ "$USE_VLLM_SERVER" == "1" ]]; then
    export LOCAL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export LOCAL_API_KEY="EMPTY"
    export USE_VLLM_SERVER USE_VLLM=0
    # vLLM is started/stopped per-stage by _ensure_vllm_running/_ensure_vllm_stopped
fi

# Sanity check — base ARGDesigner checkpoint must exist.
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: base checkpoint not found: $MODEL_DIR"
    echo "  Run pipeline_train.sh first."
    exit 1
fi

# ---- Stage 1a: Generate ARGDesigner candidates (no vLLM — GPU free) ---------
echo ""
if stage_done stage1a_gen_candidates || stage_done stage1_collect; then
    echo "▶ [Stage 1a/4] Gen candidates ... SKIPPED (already done)"
else
    echo "▶ [Stage 1a/4] Gen candidates (ARGDesigner GNN, no vLLM) ..."
    # vLLM is not running yet — ARGDesigner can use the full GPU.
    _ensure_vllm_stopped

    uv run rlhf \
        --dataset            "$DATASET" \
        --phase              gen_candidates \
        --llm_name           "$HF_MODEL" \
        --dataset_json       "$DATASET_JSON" \
        --num_tasks          "$RLHF_NUM_TASKS" \
        --preference_dir     "$PREFERENCE_DIR" \
        --min_agents         "$MIN_AGENTS" \
        --max_agents         "$MAX_AGENTS" \
        --w_correct          "$W_CORRECT" \
        --w_size             "$W_SIZE" \
        --w_edge             "$W_EDGE" \
        --pair_margin        "$PAIR_MARGIN" \
        --seed               "$SEED" \
        --sample_temperatures $SAMPLE_TEMPERATURES \
        ${ARG_MODEL_DIR:+--arg_model_dir "$ARG_MODEL_DIR"} \
        ${ARG_MODEL_SAMPLES:+--arg_model_samples "$ARG_MODEL_SAMPLES"} \
        $([ "${DIFFICULTY_FILTER}" = "1" ] && echo "--difficulty_filter") \
        $([ "${DIFFICULTY_FILTER}" = "1" ] && echo "--min_fail_rate $MIN_FAIL_RATE") \
        $([ "${WEAK_BASELINES}"    = "1" ] && echo "--weak_baselines") \
        $([ "${ROLE_SWEEP}"        = "1" ] && echo "--role_sweep") \
        $([ "${ROLE_SWEEP}"        = "1" ] && echo "--role_sweep_topology $ROLE_SWEEP_TOPOLOGY") \
        $([ "${ROLE_SWEEP}"        = "1" ] && echo "--role_sweep_n_agents $ROLE_SWEEP_N_AGENTS") \
        ${ROLE_SWEEP_MAX_COMBOS:+$([ "${ROLE_SWEEP}" = "1" ] && echo "--role_sweep_max_combos $ROLE_SWEEP_MAX_COMBOS")} \
        $(
            _dirs="${COLDSTART_DIRS:-}"
            [[ -d "$COLD_START_DIR" ]] && _dirs="${_dirs:+$_dirs }$COLD_START_DIR"
            [[ -d "$D_EFF_DIR"      ]] && _dirs="${_dirs:+$_dirs }$D_EFF_DIR"
            [[ -d "$COLD_START_DIR/rlhf_rejected" ]] && _dirs="${_dirs:+$_dirs }$COLD_START_DIR/rlhf_rejected"
            [[ -d "$D_EFF_DIR/rlhf_rejected"      ]] && _dirs="${_dirs:+$_dirs }$D_EFF_DIR/rlhf_rejected"
            [[ -n "$_dirs" ]] && echo "--coldstart_dirs $_dirs"
        )

    mark_done stage1a_gen_candidates
    echo "  ✓ Candidates → $PREFERENCE_DIR/candidates.pkl"
fi

# ---- Stage 1b: LLM scoring of candidates (vLLM must be running) -------------
echo ""
if stage_done stage1b_collect_llm || stage_done stage1_collect; then
    echo "▶ [Stage 1b/4] Collect LLM scoring ... SKIPPED (already done)"
else
    echo "▶ [Stage 1b/4] Collect LLM scoring (vLLM inference on candidates) ..."
    _ensure_vllm_running

    uv run rlhf \
        --dataset            "$DATASET" \
        --phase              collect_llm \
        --llm_name           "$HF_MODEL" \
        --dataset_json       "$DATASET_JSON" \
        --num_tasks          "$RLHF_NUM_TASKS" \
        --preference_dir     "$PREFERENCE_DIR" \
        --min_agents         "$MIN_AGENTS" \
        --max_agents         "$MAX_AGENTS" \
        --w_correct          "$W_CORRECT" \
        --w_size             "$W_SIZE" \
        --w_edge             "$W_EDGE" \
        --pair_margin        "$PAIR_MARGIN" \
        --pruning_ratio      "$PRUNING_RATIO" \
        --checkpoint_every   "$CHECKPOINT_EVERY" \
        --task_concurrency      "$TASK_CONCURRENCY" \
        --inference_concurrency "$INFERENCE_CONCURRENCY" \
        --llm_timeout           "$LLM_TIMEOUT" \
        --seed               "$SEED" \
        $(
            _dirs="${COLDSTART_DIRS:-}"
            [[ -d "$COLD_START_DIR" ]] && _dirs="${_dirs:+$_dirs }$COLD_START_DIR"
            [[ -d "$D_EFF_DIR"      ]] && _dirs="${_dirs:+$_dirs }$D_EFF_DIR"
            [[ -d "$COLD_START_DIR/rlhf_rejected" ]] && _dirs="${_dirs:+$_dirs }$COLD_START_DIR/rlhf_rejected"
            [[ -d "$D_EFF_DIR/rlhf_rejected"      ]] && _dirs="${_dirs:+$_dirs }$D_EFF_DIR/rlhf_rejected"
            [[ -n "$_dirs" ]] && echo "--coldstart_dirs $_dirs"
        )

    mark_done stage1b_collect_llm
    echo "  ✓ Collect → $PREFERENCE_DIR"
fi

# ---- Stage 2: Train reward model --------------------------------------------
echo ""
if stage_done stage2_train_rm; then
    echo "▶ [Stage 2/4] Train reward model ... SKIPPED (already done)"
else
    echo "▶ [Stage 2/4] Train reward model ..."
    _ensure_vllm_stopped

    uv run rlhf \
        --dataset          "$DATASET" \
        --phase            train_rm \
        --preference_dir   "$PREFERENCE_DIR" \
        --rm_checkpoint    "$RM_CHECKPOINT" \
        --rm_epochs        "$RM_EPOCHS" \
        --rm_lr            "$RM_LR" \
        --rm_batch_size    "$RM_BATCH_SIZE" \
        --rm_hidden_dim    "$RM_HIDDEN_DIM" \
        --rm_output_dim    "$RM_OUTPUT_DIM" \
        --rm_val_fraction  "$RM_VAL_FRACTION" \
        --both_wrong_weight  "$BOTH_WRONG_WEIGHT" \
        --both_correct_weight "$BOTH_CORRECT_WEIGHT"

    mark_done stage2_train_rm
    echo "  ✓ Reward model → $RM_CHECKPOINT"
fi

# ---- Stage 3: Train policy --------------------------------------------------
echo ""
if stage_done stage3_train_policy; then
    echo "▶ [Stage 3/4] Train RLHF policy ... SKIPPED (already done)"
else
    echo "▶ [Stage 3/4] Train RLHF policy ..."

    uv run rlhf \
        --dataset            "$DATASET" \
        --phase              train_policy \
        --llm_name           "$HF_MODEL" \
        --model_dir          "$MODEL_DIR" \
        --rm_checkpoint      "$RM_CHECKPOINT" \
        --policy_checkpoint  "$POLICY_CHECKPOINT" \
        --dataset_json       "$DATASET_JSON" \
        --num_tasks          "$POLICY_NUM_TASKS" \
        --policy_epochs      "$POLICY_EPOCHS" \
        --policy_lr          "$POLICY_LR" \
        --kl_coeff           "$KL_COEFF" \
        --samples_per_task   "$SAMPLES_PER_TASK" \
        --grad_accum_steps   "$GRAD_ACCUM_STEPS" \
        --seed               "$SEED"

    mark_done stage3_train_policy
    echo "  ✓ Policy → $POLICY_CHECKPOINT"
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

POLICY_DIR="$(dirname "$POLICY_CHECKPOINT")"
MODEL_TYPE="rlhf"
GRAPHS_ROOT="${GRAPHS_ROOT:-graphs}"
GRAPHS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${GRAPHS_ROOT}/${MODEL_TYPE}"
GRAPHS_FILE="$GRAPHS_DIR/${DATASET}_graphs.jsonl"

DECISION_METHODS=(FinalRefer FinalRefer FinalRefer FinalRefer FinalWriteCode FinalRefer)
DECISION="${DECISION_METHODS[$SLURM_ARRAY_TASK_ID]}"

LIMIT="${LIMIT:-}"
NO_EF="${NO_EF:-0}"
NO_EF_FLAG=""
[[ "$NO_EF" == "1" ]] && NO_EF_FLAG="--no_ef"

# Best-of-N flags: only pass when BEST_OF_N > 1 and required files exist
BON_FLAGS=""
if [[ "${BEST_OF_N:-1}" -gt 1 ]]; then
    ROLE_EMB_FILE="$COLD_START_DIR/precomputed_role_embeddings.pkl"
    if [[ ! -f "$RM_CHECKPOINT" ]]; then
        echo "WARNING: BEST_OF_N=$BEST_OF_N but RM checkpoint not found: $RM_CHECKPOINT — disabling BoN"
    elif [[ ! -f "$ROLE_EMB_FILE" ]]; then
        echo "WARNING: BEST_OF_N=$BEST_OF_N but role embeddings not found: $ROLE_EMB_FILE — disabling BoN"
    else
        BON_FLAGS="--best_of_n $BEST_OF_N --bon_temperature $BON_TEMPERATURE --rm_checkpoint $RM_CHECKPOINT --role_emb_path $ROLE_EMB_FILE"
        echo "  [BoN] Enabled: N=$BEST_OF_N  temperature=$BON_TEMPERATURE"
    fi
fi

export PYTHONPATH

# ---- Stage 4a: Generate graphs (RLHF policy) --------------------------------
echo ""
if stage_done stage4a_gen_graphs; then
    echo "▶ [Stage 4a/4] Generate graphs ... SKIPPED (already done)"
else
    echo "▶ [Stage 4a/4] Generate graphs (RLHF policy) ..."

    mkdir -p "$GRAPHS_DIR"

    uv run python experiment/generate_graphs.py \
        --model_path   "$POLICY_DIR" \
        --dataset      "$DATASET" \
        --dataset_path "$DATASET_JSON" \
        --output_file  "$GRAPHS_FILE" \
        --model_type   "$MODEL_TYPE" \
        ${TASK_SPLIT_RAW:+--task_split_path "$PROJECT_ROOT/$TASK_SPLIT_RAW"} \
        ${LIMIT:+--limit "$LIMIT"} \
        $NO_EF_FLAG \
        $BON_FLAGS

    mark_done stage4a_gen_graphs
    echo "  ✓ Graphs → $GRAPHS_FILE"
fi

# ---- Stage 4b: Benchmark pre-generated graphs (RLHF policy) -----------------
echo ""
if stage_done stage4b_benchmark; then
    echo "▶ [Stage 4b/4] Benchmark ... SKIPPED (already done)"
else
    echo "▶ [Stage 4b/4] Benchmark (RLHF pre-generated graphs) ..."
    _ensure_vllm_running

    RESULTS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/rlhf"
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
echo ""
TRAIN_GRAPHS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${GRAPHS_ROOT}/rlhf_train"
TRAIN_GRAPHS_FILE="$TRAIN_GRAPHS_DIR/${DATASET}_graphs.jsonl"

if stage_done stage5a_gen_graphs_train; then
    echo "▶ [Stage 5a/5] Generate train-set graphs ... SKIPPED (already done)"
else
    echo "▶ [Stage 5a/5] Generate train-set graphs (overfitting check) ..."
    mkdir -p "$TRAIN_GRAPHS_DIR"

    uv run python experiment/generate_graphs.py \
        --model_path   "$POLICY_DIR" \
        --dataset      "$DATASET" \
        --dataset_path "$DATASET_JSON" \
        --output_file  "$TRAIN_GRAPHS_FILE" \
        --model_type   rlhf_train \
        --split_key    "base_tasks_indices,finetune_tasks_indices" \
        ${TASK_SPLIT_RAW:+--task_split_path "$PROJECT_ROOT/$TASK_SPLIT_RAW"} \
        $NO_EF_FLAG \
        $BON_FLAGS

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

    TRAIN_RESULTS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/rlhf_train"
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
        --model_type       rlhf_train

    mark_done stage5b_benchmark_train
    echo "  ✓ Train results → $TRAIN_OUTPUT_FILE"
fi

echo ""
echo "════════════════════════════════════════════════"
echo "  RLHF pipeline complete for $DATASET"
echo "  Test  results : ${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/rlhf/summary.jsonl"
echo "  Train results : ${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/rlhf_train/summary.jsonl"
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════"