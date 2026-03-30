#!/bin/bash
#SBATCH --job-name=arg_pipeline_rlhf
#SBATCH --output=logs/pipeline_rlhf_%A_%a.out
#SBATCH --error=logs/pipeline_rlhf_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5          # 0=gsm8k 1=aqua 2=multiarith 3=svamp 4=humaneval 5=mmlu
#SBATCH -p gpu
#SBATCH --time=2-00:00:00
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
#   CHECKPOINT_ROOT    Phase-1/2 checkpoint sub-dir     (default: checkpoints)
#   RLHF_NUM_TASKS     tasks to sample for collect      (default: 100)
#   PREFERENCE_ROOT    preference data sub-dir           (default: rlhf_data)
#   RM_ROOT            reward model checkpoint sub-dir   (default: rlhf_checkpoints)
#   POLICY_ROOT        policy checkpoint sub-dir         (default: rlhf_checkpoints)
#   COLDSTART_DIRS     space-separated .pt dirs for free pairs (optional)
#   ARG_MODEL_SAMPLES  unique ARGDesigner graphs per task (optional)
#   EVAL_BATCH         benchmark batch size              (default: 8)
#   RESULTS_ROOT       results sub-dir                  (default: benchmark_results)
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
MODEL_SLUG="${HF_MODEL//\//-}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
MODEL_SLUG="${MODEL_SLUG}-vllm-$([ "${DISABLE_THINKING}" = "1" ] && echo no_thinking || echo thinking)"
export DISABLE_THINKING PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SEED="${SEED:-42}"

# Collect
RLHF_NUM_TASKS="${RLHF_NUM_TASKS:-100}"
PREFERENCE_ROOT="${PREFERENCE_ROOT:-rlhf_data}"
MIN_AGENTS="${MIN_AGENTS:-2}"
W_CORRECT="${W_CORRECT:-0.6}"
W_SIZE="${W_SIZE:-0.2}"
W_TOKEN="${W_TOKEN:-0.2}"
PAIR_MARGIN="${PAIR_MARGIN:-0.05}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2}"
LLM_TIMEOUT="${LLM_TIMEOUT:-1200}"
SAMPLE_TEMPERATURES="${SAMPLE_TEMPERATURES:-1.0 1.5 2.0}"
ARG_MODEL_DIR="${ARG_MODEL_DIR:-}"
COLDSTART_DIRS="${COLDSTART_DIRS:-}"
ARG_MODEL_SAMPLES="${ARG_MODEL_SAMPLES:-}"

# Reward model
RM_ROOT="${RM_ROOT:-rlhf_checkpoints}"
RM_EPOCHS="${RM_EPOCHS:-20}"
RM_LR="${RM_LR:-1e-4}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-32}"
RM_HIDDEN_DIM="${RM_HIDDEN_DIM:-256}"
RM_OUTPUT_DIM="${RM_OUTPUT_DIM:-128}"
RM_VAL_FRACTION="${RM_VAL_FRACTION:-0.1}"
BOTH_WRONG_WEIGHT="${BOTH_WRONG_WEIGHT:-0.2}"

# Policy
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
POLICY_ROOT="${POLICY_ROOT:-rlhf_checkpoints}"
POLICY_EPOCHS="${POLICY_EPOCHS:-30}"
POLICY_NUM_TASKS="${POLICY_NUM_TASKS:-200}"
POLICY_LR="${POLICY_LR:-5e-6}"
KL_COEFF="${KL_COEFF:-0.1}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-2}"

# Benchmark
EVAL_BATCH="${EVAL_BATCH:-2}"
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_results}"

# Derived paths
PREFERENCE_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${PREFERENCE_ROOT}/${DATASET}"
RM_CHECKPOINT="$PROJECT_ROOT/${MODEL_SLUG}/${RM_ROOT}/${DATASET}/reward_model.pth"
POLICY_CHECKPOINT="$PROJECT_ROOT/${MODEL_SLUG}/${POLICY_ROOT}/${DATASET}/policy_rlhf.pth"
MODEL_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${CHECKPOINT_ROOT}/${DATASET}"

# ---- vLLM inference server --------------------------------------------------
# Set USE_VLLM_SERVER=0 to disable and fall back to HuggingFace transformers.
USE_VLLM_SERVER="${USE_VLLM_SERVER:-1}"
VLLM_PORT="${VLLM_PORT:-$((6789 + ${SLURM_ARRAY_TASK_ID:-0}))}"
VLLM_TP="${VLLM_TP:-1}"                  # tensor-parallel GPUs for the server
VLLM_SERVE_DIR="${VLLM_SERVE_DIR:-/home/users/psuwannapichat/work_space/vllm_temp}"
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-${VLLM_SERVE_DIR}/qwen3_nonthinking.jinja}"
VLLM_PID=""

_start_vllm_server() {
    echo "▶ Starting vLLM server for '$HF_MODEL' on port $VLLM_PORT ..."
    mkdir -p "$PROJECT_ROOT/logs"
    ("$VLLM_SERVE_DIR/.venv/bin/vllm" serve "$HF_MODEL" \
        --port                   "$VLLM_PORT" \
        --dtype                  float16 \
        --trust-remote-code \
        --max-model-len          8192 \
        --gpu-memory-utilization 0.9 \
        --tensor-parallel-size   "$VLLM_TP" \
        --enforce-eager \
        --chat-template          "$VLLM_CHAT_TEMPLATE") \
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
    [[ -n "${VLLM_PID:-}" ]] && kill "$VLLM_PID" 2>/dev/null && echo "  ✓ vLLM server stopped."
}
trap _stop_vllm_server EXIT

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
echo "  Started at       : $(date)"
echo "════════════════════════════════════════════════"

if [[ "$USE_VLLM_SERVER" == "1" ]]; then
    export LOCAL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export LOCAL_API_KEY="EMPTY"
    export USE_VLLM_SERVER USE_VLLM=0
    _start_vllm_server
    echo ""
fi

# Sanity check — base ARGDesigner checkpoint must exist.
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: base checkpoint not found: $MODEL_DIR"
    echo "  Run pipeline_train.sh first."
    exit 1
fi

# ---- Stage 1: Collect preference pairs --------------------------------------
echo ""
if stage_done stage1_collect; then
    echo "▶ [Stage 1/4] Collect preference pairs ... SKIPPED (already done)"
else
    echo "▶ [Stage 1/4] Collect preference pairs ..."

    uv run rlhf \
        --dataset            "$DATASET" \
        --phase              collect \
        --llm_name           "$HF_MODEL" \
        --dataset_json       "$DATASET_JSON" \
        --num_tasks          "$RLHF_NUM_TASKS" \
        --preference_dir     "$PREFERENCE_DIR" \
        --min_agents         "$MIN_AGENTS" \
        --max_agents         "$MAX_AGENTS" \
        --w_correct          "$W_CORRECT" \
        --w_size             "$W_SIZE" \
        --w_token            "$W_TOKEN" \
        --pair_margin        "$PAIR_MARGIN" \
        --checkpoint_every   "$CHECKPOINT_EVERY" \
        --llm_timeout        "$LLM_TIMEOUT" \
        --seed               "$SEED" \
        --sample_temperatures $SAMPLE_TEMPERATURES \
        ${ARG_MODEL_DIR:+--arg_model_dir "$ARG_MODEL_DIR"} \
        ${COLDSTART_DIRS:+--coldstart_dirs $COLDSTART_DIRS} \
        ${ARG_MODEL_SAMPLES:+--arg_model_samples "$ARG_MODEL_SAMPLES"}

    mark_done stage1_collect
    echo "  ✓ Collect → $PREFERENCE_DIR"
fi

# ---- Stage 2: Train reward model --------------------------------------------
echo ""
if stage_done stage2_train_rm; then
    echo "▶ [Stage 2/4] Train reward model ... SKIPPED (already done)"
else
    echo "▶ [Stage 2/4] Train reward model ..."

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
        --both_wrong_weight "$BOTH_WRONG_WEIGHT"

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
        $NO_EF_FLAG

    mark_done stage4a_gen_graphs
    echo "  ✓ Graphs → $GRAPHS_FILE"
fi

# ---- Stage 4b: Benchmark pre-generated graphs (RLHF policy) -----------------
echo ""
if stage_done stage4b_benchmark; then
    echo "▶ [Stage 4b/4] Benchmark ... SKIPPED (already done)"
else
    echo "▶ [Stage 4b/4] Benchmark (RLHF pre-generated graphs) ..."

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

echo ""
echo "════════════════════════════════════════════════"
echo "  RLHF pipeline complete for $DATASET"
echo "  Summary : ${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/rlhf/summary.jsonl"
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════"
