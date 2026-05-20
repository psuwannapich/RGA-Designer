#!/bin/bash
#SBATCH --job-name=global_rm
#SBATCH --output=logs/global_rm_%j.out
#SBATCH --error=logs/global_rm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --time=16:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# Global reward model pipeline.
#
# Pools preference pairs already collected by pipeline_rlhf_exp.sh and trains
# ONE reward model across all INCLUDED datasets, then fine-tunes a per-dataset
# ARGDesigner policy using that global RM.
#
# EXCLUDE_DATASETS: datasets whose preference pairs are WITHHELD from RM
# training.  They still get a policy and benchmark — this tests whether the
# global RM generalises to task types it has never seen.
#
# Prerequisite:
#   pipeline_arg.sh must have completed          → ARG checkpoints
#   pipeline_rlhf_exp.sh stages 1a+1b completed  → preference pairs
#
# Reads  from:
#   <llm_slug>/arg/<arg_run>/checkpoints/<dataset>/   (policy init)
#   <llm_slug>/<src_exp_name>/<src_run>/rlhf_data/<dataset>/  (preference pairs)
#
# Writes to:
#   <llm_slug>/<exp_name>/<run>/
#     global_rm/reward_model.pth
#     rlhf_checkpoints/<dataset>/policy_rlhf.pth
#     graphs/rlhf/<dataset>_graphs.jsonl
#     graphs/rlhf_train/<dataset>_graphs.jsonl
#     benchmark_results/pregraph/rlhf/<dataset>.jsonl
#     benchmark_results/pregraph/rlhf_train/<dataset>.jsonl
#     state/
#
# Usage:
#   sbatch slurm/pipeline_global_rm_exp.sh
#
#   # Leave gsm8k + aqua out of RM training (generalization test):
#   EXCLUDE_DATASETS="gsm8k aqua" sbatch slurm/pipeline_global_rm_exp.sh
#
#   # Use a specific source experiment and run index:
#   SRC_EXP_NAME=kl05 SRC_RUN=2 EXP_NAME=global_rm_kl05 RUN=2 \
#       sbatch slurm/pipeline_global_rm_exp.sh
#
# Optional env vars:
#   HF_MODEL            HuggingFace model ID                   (default: Qwen/Qwen3-4B)
#   LLM_SLUG            directory name for the LLM             (default: HF_MODEL with / → -)
#   SRC_EXP_NAME        experiment to read preference data from (default: baseline)
#   SRC_RUN             run index of the source experiment      (default: 0)
#   EXP_NAME            name for this global RM experiment      (default: global_rm)
#   RUN                 repetition index 0-4                    (default: 0)
#   ARG_RUN             which ARG run to use for policy init    (default: 0)
#   EXCLUDE_DATASETS    space-separated datasets to exclude from RM training
#                       (still evaluated; default: "")
#   PORT_OFFSET         add to vLLM port to avoid node collisions (default: 0)
#   DISABLE_THINKING    1 = no-thinking mode (Qwen3)            (default: 1)
#   BOTH_WRONG_WEIGHT   RM loss weight for both-wrong pairs      (default: 0)
#   BOTH_CORRECT_WEIGHT RM loss weight for both-correct pairs   (default: 0.1)
#   RM_LOSS             RM training loss: bradley_terry|bce      (default: bradley_terry)
#   RM_EPOCHS           reward model training epochs            (default: 30)
#   RM_LR               reward model learning rate              (default: 1e-4)
#   POLICY_NUM_TASKS    tasks for policy training per dataset   (default: 100)
#   POLICY_EPOCHS       policy fine-tune epochs                 (default: 30)
#   POLICY_LR           policy learning rate                    (default: 5e-6)
#   KL_COEFF            KL penalty coefficient                  (default: 0.5)
#   LAMBDA_EFF          efficiency bonus weight added to RM reward in GRPO
#                       bonus = lambda_eff * mean(1-nodes/max, 1-edges/max_edges)
#                       (default: 0 = disabled, recommended 0.1-0.3)
#   SAMPLES_PER_TASK    graphs sampled per task (GRPO)          (default: 4)
#   BEST_OF_N           BoN graph selection at inference        (default: 5)
#   EVAL_BATCH          benchmark batch size                    (default: 8)
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
DATASET_MAX_AGENTS=(4 4 4 4 5 6)
DECISION_METHODS=(FinalRefer FinalRefer FinalRefer FinalRefer FinalWriteCode FinalRefer)

# ---- Model configuration ----------------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"

if [[ "$HF_MODEL" == *"Llama"* ]] || [[ "$HF_MODEL" == *"llama"* ]]; then
    IS_LLAMA=1
else
    IS_LLAMA=0
fi

DISABLE_THINKING="${DISABLE_THINKING:-$([ "$IS_LLAMA" = "1" ] && echo 0 || echo 1)}"
LLM_SLUG="${LLM_SLUG:-${HF_MODEL//\//-}}"

# This experiment
EXP_NAME="${EXP_NAME:-global_rm}"
RUN="${RUN:-0}"
ARG_RUN="${RUN:-0}"

# Source experiment — where to read already-collected preference pairs from
SRC_EXP_NAME="${SRC_EXP_NAME:-baseline}"
SRC_RUN="${RUN:-0}"

# Datasets withheld from RM training (space-separated, e.g. "gsm8k humaneval")
EXCLUDE_DATASETS="${EXCLUDE_DATASETS:-}"

export DISABLE_THINKING LLM_SLUG EXP_NAME PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---- Hyperparameters --------------------------------------------------------
SEED="${SEED:-$((42 + RUN))}"

# Reward model
RM_EPOCHS="${RM_EPOCHS:-30}"
RM_LR="${RM_LR:-1e-4}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-32}"
RM_HIDDEN_DIM="${RM_HIDDEN_DIM:-256}"
RM_OUTPUT_DIM="${RM_OUTPUT_DIM:-128}"
RM_VAL_FRACTION="${RM_VAL_FRACTION:-0.1}"
BOTH_WRONG_WEIGHT="${BOTH_WRONG_WEIGHT:-0}"
BOTH_CORRECT_WEIGHT="${BOTH_CORRECT_WEIGHT:-0.1}"
RM_LOSS="${RM_LOSS:-bradley_terry}"

# Policy
POLICY_NUM_TASKS="${POLICY_NUM_TASKS:-100}"
POLICY_EPOCHS="${POLICY_EPOCHS:-30}"
POLICY_LR="${POLICY_LR:-5e-6}"
KL_COEFF="${KL_COEFF:-0.5}"
LAMBDA_EFF="${LAMBDA_EFF:-0.0}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"

# Benchmark / graph generation
EVAL_BATCH="${EVAL_BATCH:-8}"
LIMIT="${LIMIT:-}"
NO_EF="${NO_EF:-0}"
BEST_OF_N="${BEST_OF_N:-1}"
BON_TEMPERATURE="${BON_TEMPERATURE:-1}"

# ---- Directory layout -------------------------------------------------------
SRC_BASE="$PROJECT_ROOT/${LLM_SLUG}/${SRC_EXP_NAME}/${SRC_RUN}"
ARG_BASE="$PROJECT_ROOT/${LLM_SLUG}/arg/${ARG_RUN}"
RLHF_BASE="$PROJECT_ROOT/${LLM_SLUG}/${EXP_NAME}/${RUN}"

GLOBAL_RM_CHECKPOINT="$RLHF_BASE/global_rm/reward_model.pth"

# ---- vLLM inference server --------------------------------------------------
USE_VLLM_SERVER="${USE_VLLM_SERVER:-1}"
PORT_OFFSET="${PORT_OFFSET:-0}"
VLLM_PORT="${VLLM_PORT:-$((9700 + RUN * 10 + PORT_OFFSET))}"
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
    echo "  ✓ vLLM server stopped."
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
            kill -9 "${VLLM_PID}" 2>/dev/null || true
            break
        fi
    done
    VLLM_PID=""
    echo "  ✓ vLLM server stopped. Waiting 5s for GPU VRAM to be reclaimed ..."
    sleep 5
}

# ---- Stage checkpointing ----------------------------------------------------
# stage_done / mark_done: global scope (uses STATE_DIR explicitly set by caller).
STATE_DIR="$RLHF_BASE/state"
stage_done() { [[ -f "$STATE_DIR/$1.done" ]]; }
mark_done()  { touch "$STATE_DIR/$1.done"; echo "  ✔ checkpoint: $STATE_DIR/$1.done"; }

# ds_done / ds_mark: per-dataset scope (uses DATASET loop variable directly,
# never relies on the mutable STATE_DIR global).
ds_done() { [[ -f "$RLHF_BASE/state/${DATASET}/$1.done" ]]; }
ds_mark() { local _d="$RLHF_BASE/state/${DATASET}"; mkdir -p "$_d"; touch "$_d/$1.done"; echo "  ✔ checkpoint: $_d/$1.done"; }

# ---- Build included / excluded dataset lists --------------------------------
INCLUDED_PREF_DIRS=()
INCLUDED_DATASETS=()
EXCLUDED_DATASETS_DISPLAY=()

for ds in "${DATASETS[@]}"; do
    pref_dir="$SRC_BASE/rlhf_data/${ds}"
    if [[ " $EXCLUDE_DATASETS " == *" $ds "* ]]; then
        EXCLUDED_DATASETS_DISPLAY+=("$ds")
    elif [[ -d "$pref_dir" ]]; then
        INCLUDED_PREF_DIRS+=("$pref_dir")
        INCLUDED_DATASETS+=("$ds")
    else
        echo "  WARNING: no preference data for $ds at $pref_dir — skipping from RM training"
        EXCLUDED_DATASETS_DISPLAY+=("$ds [missing data]")
    fi
done

if [[ ${#INCLUDED_PREF_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: no preference data found in $SRC_BASE/rlhf_data/." >&2
    echo "  Run pipeline_rlhf_exp.sh stages 1a+1b first (SRC_EXP_NAME=$SRC_EXP_NAME, SRC_RUN=$SRC_RUN)." >&2
    exit 1
fi

if [[ ! -f "$ARG_BASE/checkpoints/${DATASETS[0]}/ef_best_model.pth" ]]; then
    echo "ERROR: ARG checkpoint not found under $ARG_BASE/checkpoints/"
    echo "  Run pipeline_arg.sh first (LLM_SLUG=$LLM_SLUG, ARG_RUN=$ARG_RUN)."
    exit 1
fi

mkdir -p "$STATE_DIR/global"

# ---- Header -----------------------------------------------------------------
echo "════════════════════════════════════════════════"
echo "  Global Reward Model Pipeline"
echo "  Job ${SLURM_JOB_ID:-local}"
echo "  Node              : ${SLURM_NODELIST:-local}"
echo "  Model             : $HF_MODEL"
echo "  LLM slug          : $LLM_SLUG"
echo "  Experiment        : $EXP_NAME  (run $RUN)"
echo "  Source exp        : ${LLM_SLUG}/${SRC_EXP_NAME}/${SRC_RUN}/"
echo "  ARG base          : ${LLM_SLUG}/arg/${ARG_RUN}/"
echo "  Output base       : ${LLM_SLUG}/${EXP_NAME}/${RUN}/"
echo "  RM training data  : ${INCLUDED_DATASETS[*]:-none}"
echo "  Excluded datasets : ${EXCLUDED_DATASETS_DISPLAY[*]:-none}"
echo "  RM epochs         : $RM_EPOCHS  lr=$RM_LR"
echo "  Policy epochs     : $POLICY_EPOCHS  lr=$POLICY_LR  kl=$KL_COEFF  lambda_eff=$LAMBDA_EFF"
echo "  Best-of-N         : $([ "${BEST_OF_N:-1}" -gt 1 ] && echo "N=$BEST_OF_N  temp=$BON_TEMPERATURE" || echo "disabled")"
echo "  vLLM server       : $([ "$USE_VLLM_SERVER" = "1" ] && echo "enabled (port $VLLM_PORT, tp=$VLLM_TP)" || echo "disabled (HF backend)")"
echo "  Started at        : $(date)"
echo "════════════════════════════════════════════════"

if [[ "$USE_VLLM_SERVER" == "1" ]]; then
    export LOCAL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export LOCAL_API_KEY="EMPTY"
    export USE_VLLM_SERVER USE_VLLM=0
fi

# ============================================================================
# Stage 1 — Train global reward model
# ============================================================================
echo ""
STATE_DIR="$RLHF_BASE/state/global"
mkdir -p "$STATE_DIR"

if stage_done stage1_train_global_rm; then
    echo "▶ [Stage 1] Train global RM ... SKIPPED (already done)"
else
    echo "▶ [Stage 1] Training global reward model on ${#INCLUDED_PREF_DIRS[@]} dataset(s) ..."
    for d in "${INCLUDED_PREF_DIRS[@]}"; do echo "  + $d"; done
    _ensure_vllm_stopped

    mkdir -p "$(dirname "$GLOBAL_RM_CHECKPOINT")"

    uv run rlhf \
        --dataset             gsm8k \
        --phase               train_rm \
        --preference_dirs     "${INCLUDED_PREF_DIRS[@]}" \
        --rm_checkpoint       "$GLOBAL_RM_CHECKPOINT" \
        --rm_epochs           "$RM_EPOCHS" \
        --rm_lr               "$RM_LR" \
        --rm_batch_size       "$RM_BATCH_SIZE" \
        --rm_hidden_dim       "$RM_HIDDEN_DIM" \
        --rm_output_dim       "$RM_OUTPUT_DIM" \
        --rm_val_fraction     "$RM_VAL_FRACTION" \
        --both_wrong_weight   "$BOTH_WRONG_WEIGHT" \
        --both_correct_weight "$BOTH_CORRECT_WEIGHT" \
        --rm_loss             "$RM_LOSS"

    mark_done stage1_train_global_rm
    echo "  ✓ Global RM → $GLOBAL_RM_CHECKPOINT"
fi

# ============================================================================
# Stages 2-4 — Two passes so vLLM starts only once
#
#   Pass 1 (no vLLM) : stage2_train_policy + stage3a_gen_graphs +
#                      stage4a_gen_graphs_train   (all datasets)
#   Pass 2 (vLLM on) : stage3b_benchmark + stage4b_benchmark_train
#                                                 (all datasets)
# ============================================================================
NO_EF_FLAG=""
[[ "$NO_EF" == "1" ]] && NO_EF_FLAG="--no_ef"

# ---- Pass 1 (no vLLM): policy training + all graph generation ---------------
echo ""
echo "════════════ Pass 1: Policy Training + Graph Generation (all datasets) ════════════"
_ensure_vllm_stopped

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$i]}"
    TASK_SPLIT_RAW="${TASK_SPLIT_PATHS[$i]}"

    IS_EXCLUDED=0
    [[ " $EXCLUDE_DATASETS " == *" $DATASET "* ]] && IS_EXCLUDED=1
    EXCLUDED_TAG=""
    [[ "$IS_EXCLUDED" = "1" ]] && EXCLUDED_TAG=" [excluded from RM — generalization test]"

    ARG_CHECKPOINT_DIR="$ARG_BASE/checkpoints/${DATASET}"
    ARG_COLD_START_DIR="$ARG_BASE/ColdStartData/${DATASET}"
    POLICY_CHECKPOINT="$RLHF_BASE/rlhf_checkpoints/${DATASET}/policy_rlhf.pth"
    POLICY_DIR="$(dirname "$POLICY_CHECKPOINT")"
    GRAPHS_DIR="$RLHF_BASE/graphs/rlhf"
    GRAPHS_FILE="$GRAPHS_DIR/${DATASET}_graphs.jsonl"
    TRAIN_GRAPHS_DIR="$RLHF_BASE/graphs/rlhf_train"
    TRAIN_GRAPHS_FILE="$TRAIN_GRAPHS_DIR/${DATASET}_graphs.jsonl"

    STATE_DIR="$RLHF_BASE/state/${DATASET}"
    mkdir -p "$STATE_DIR"

    echo ""
    echo "  ── $DATASET${EXCLUDED_TAG}"

    if [[ ! -f "$ARG_CHECKPOINT_DIR/ef_best_model.pth" ]]; then
        echo "  WARNING: ARG checkpoint missing for $DATASET — skipping"
        continue
    fi

    # ---- Stage 2: Train policy with global RM --------------------------------
    if stage_done stage2_train_policy; then
        echo "▶ [Stage 2] Train policy ... SKIPPED (already done)"
    else
        echo "▶ [Stage 2] Train policy ($DATASET) using global RM ..."

        uv run rlhf \
            --dataset            "$DATASET" \
            --phase              train_policy \
            --llm_name           "$HF_MODEL" \
            --model_dir          "$ARG_CHECKPOINT_DIR" \
            --rm_checkpoint      "$GLOBAL_RM_CHECKPOINT" \
            --policy_checkpoint  "$POLICY_CHECKPOINT" \
            --dataset_json       "$DATASET_JSON" \
            --num_tasks          "$POLICY_NUM_TASKS" \
            --policy_epochs      "$POLICY_EPOCHS" \
            --policy_lr          "$POLICY_LR" \
            --kl_coeff           "$KL_COEFF" \
            --lambda_eff         "$LAMBDA_EFF" \
            --samples_per_task   "$SAMPLES_PER_TASK" \
            --grad_accum_steps   "$GRAD_ACCUM_STEPS" \
            --seed               "$SEED"

        mark_done stage2_train_policy
        echo "  ✓ Policy → $POLICY_CHECKPOINT"
    fi

    # Compute BoN flags once for both graph generation stages
    BON_FLAGS=""
    if [[ "${BEST_OF_N:-1}" -gt 1 ]]; then
        ROLE_EMB_FILE="$ARG_COLD_START_DIR/precomputed_role_embeddings.pkl"
        if [[ -f "$GLOBAL_RM_CHECKPOINT" && -f "$ROLE_EMB_FILE" ]]; then
            BON_FLAGS="--best_of_n $BEST_OF_N --bon_temperature $BON_TEMPERATURE --rm_checkpoint $GLOBAL_RM_CHECKPOINT --role_emb_path $ROLE_EMB_FILE"
        else
            echo "  [BoN] Skipped (missing RM checkpoint or role embeddings for $DATASET)"
        fi
    fi

    # ---- Stage 3a: Generate test-set graphs ----------------------------------
    if stage_done stage3a_gen_graphs; then
        echo "▶ [Stage 3a] Generate test graphs ... SKIPPED (already done)"
    else
        echo "▶ [Stage 3a] Generate test-set graphs ($DATASET) ..."
        mkdir -p "$GRAPHS_DIR"

        uv run python experiment/generate_graphs.py \
            --model_path   "$POLICY_DIR" \
            --dataset      "$DATASET" \
            --dataset_path "$DATASET_JSON" \
            --output_file  "$GRAPHS_FILE" \
            --model_type   rlhf \
            ${TASK_SPLIT_RAW:+--task_split_path "$PROJECT_ROOT/$TASK_SPLIT_RAW"} \
            ${LIMIT:+--limit "$LIMIT"} \
            $NO_EF_FLAG \
            $BON_FLAGS

        mark_done stage3a_gen_graphs
        echo "  ✓ Graphs → $GRAPHS_FILE"
    fi

    # ---- Stage 4a: Generate train-set graphs (overfitting check) -------------
    if stage_done stage4a_gen_graphs_train; then
        echo "▶ [Stage 4a] Generate train-set graphs ... SKIPPED (already done)"
    else
        echo "▶ [Stage 4a] Generate train-set graphs ($DATASET) ..."
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

        mark_done stage4a_gen_graphs_train
        echo "  ✓ Train graphs → $TRAIN_GRAPHS_FILE"
    fi
done

# ---- Pass 2 (vLLM on): all benchmarks ---------------------------------------
echo ""
echo "════════════ Pass 2: Benchmark (all datasets) ════════════"
_ensure_vllm_running

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    DECISION="${DECISION_METHODS[$i]}"

    IS_EXCLUDED=0
    [[ " $EXCLUDE_DATASETS " == *" $DATASET "* ]] && IS_EXCLUDED=1
    EXCLUDED_TAG=""
    [[ "$IS_EXCLUDED" = "1" ]] && EXCLUDED_TAG=" [excluded from RM — generalization test]"

    GRAPHS_FILE="$RLHF_BASE/graphs/rlhf/${DATASET}_graphs.jsonl"
    RESULTS_DIR="$RLHF_BASE/benchmark_results/pregraph/rlhf"
    TRAIN_GRAPHS_FILE="$RLHF_BASE/graphs/rlhf_train/${DATASET}_graphs.jsonl"
    TRAIN_RESULTS_DIR="$RLHF_BASE/benchmark_results/pregraph/rlhf_train"

    STATE_DIR="$RLHF_BASE/state/${DATASET}"

    echo ""
    echo "  ── $DATASET${EXCLUDED_TAG}"

    # ---- Stage 3b: Benchmark test set ----------------------------------------
    if stage_done stage3b_benchmark; then
        echo "▶ [Stage 3b] Benchmark test set ... SKIPPED (already done)"
    elif [[ ! -f "$GRAPHS_FILE" ]]; then
        echo "  WARNING: graphs file missing for $DATASET — skipping test benchmark"
    else
        echo "▶ [Stage 3b] Benchmark test set ($DATASET) ..."

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
            --model_type       rlhf \
            ${LIMIT:+--limit "$LIMIT"}

        mark_done stage3b_benchmark
        echo "  ✓ Results → $OUTPUT_FILE"
    fi

    # ---- Stage 4b: Benchmark train set ----------------------------------------
    if stage_done stage4b_benchmark_train; then
        echo "▶ [Stage 4b] Benchmark train set ... SKIPPED (already done)"
    elif [[ ! -f "$TRAIN_GRAPHS_FILE" ]]; then
        echo "  WARNING: train graphs file missing for $DATASET — skipping train benchmark"
    else
        echo "▶ [Stage 4b] Benchmark train set ($DATASET) ..."

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

        mark_done stage4b_benchmark_train
        echo "  ✓ Train results → $TRAIN_OUTPUT_FILE"
    fi
done

echo ""
echo "════════════════════════════════════════════════"
echo "  Global RM pipeline complete"
echo "  Experiment      : ${LLM_SLUG}/${EXP_NAME}/${RUN}/"
echo "  Global RM       : ${LLM_SLUG}/${EXP_NAME}/${RUN}/global_rm/reward_model.pth"
echo "  RM trained on   : ${INCLUDED_DATASETS[*]:-none}"
echo "  RM excluded     : ${EXCLUDED_DATASETS_DISPLAY[*]:-none}"
echo "  Test  results   : ${LLM_SLUG}/${EXP_NAME}/${RUN}/benchmark_results/pregraph/rlhf/summary.jsonl"
echo "  Train results   : ${LLM_SLUG}/${EXP_NAME}/${RUN}/benchmark_results/pregraph/rlhf_train/summary.jsonl"
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════"
