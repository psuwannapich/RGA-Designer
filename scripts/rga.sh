#!/bin/bash
# ---------------------------------------------------------------------------
# RLHF pipeline — collect → train_rm → train_policy → benchmark
#
# Prerequisites: scripts/train.sh must have completed so that
#   ${MODEL_SLUG}/checkpoints/${DATASET}/ef_best_model.pth exists.
#
# Usage:
#   DATASET_IDX=0 bash scripts/rga.sh          # gsm8k
#   DATASET=gsm8k bash scripts/rga.sh          # by name
#   bash scripts/rga.sh 0                      # positional arg
#
# Dataset index mapping:
#   0=gsm8k  1=aqua  2=multiarith  3=svamp  4=humaneval  5=mmlu
#
# Key environment variables (all optional):
#   HF_MODEL         HuggingFace model ID or name passed to --llm_name
#                    (default: Qwen/Qwen3-4B)
#   LOCAL_BASE_URL   OpenAI-compatible API base URL.
#                    Set this to use a vLLM server, Ollama, or a commercial API.
#                    Leave unset to use the HuggingFace transformers backend.
#   LOCAL_API_KEY    API key for the above endpoint (default: EMPTY)
#   DISABLE_THINKING 1 = Qwen3 no-thinking mode (default: 1)
#   RLHF_NUM_TASKS   tasks for preference collection (default: 100)
#   KL_COEFF         KL penalty for policy training (default: 0.2)
#   BEST_OF_N        BoN candidates at inference (default: 5)
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

# ---- LLM backend ------------------------------------------------------------
# Set LOCAL_BASE_URL to use an OpenAI-compatible API (local vLLM, Ollama, or
# a commercial provider). Leave unset to fall back to HuggingFace transformers.
if [[ -n "${LOCAL_BASE_URL:-}" ]]; then
    export LOCAL_BASE_URL LOCAL_API_KEY="${LOCAL_API_KEY:-EMPTY}"
    export USE_VLLM_SERVER=1 USE_VLLM=0
else
    export USE_VLLM_SERVER=0 USE_VLLM=0
fi

# ---- Hyperparameters --------------------------------------------------------
SEED="${SEED:-42}"

# Collect
RLHF_NUM_TASKS="${RLHF_NUM_TASKS:-100}"
PREFERENCE_ROOT="${PREFERENCE_ROOT:-rga_data}"
MIN_AGENTS="${MIN_AGENTS:-2}"
W_CORRECT="${W_CORRECT:-0.6}"
# Split reward (RGA-Designer): the correctness model predicts task completion only, and
# graph size enters at scoring time with lambda_size in GRPO and Best-of-N.
RM_LOSS="${RM_LOSS:-per_graph_bce}"
LAMBDA_EFF="${LAMBDA_EFF:-0.6}"
SIZE_LAMBDA="${SIZE_LAMBDA:-0.6}"
REWARD_SQUASH="${REWARD_SQUASH:-1}"
SQUASH_FLAG=""; [[ "$REWARD_SQUASH" == "1" ]] && SQUASH_FLAG="--reward_squash"
# No size term in the collected preference labels (it enters at scoring time instead).
W_SIZE="${W_SIZE:-0}"
W_EDGE="${W_EDGE:-0}"
PAIR_MARGIN="${PAIR_MARGIN:-0.05}"
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
RM_ROOT="${RM_ROOT:-rga_checkpoints}"
RM_EPOCHS="${RM_EPOCHS:-30}"
RM_LR="${RM_LR:-1e-4}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-32}"
RM_HIDDEN_DIM="${RM_HIDDEN_DIM:-256}"
RM_OUTPUT_DIM="${RM_OUTPUT_DIM:-128}"
RM_VAL_FRACTION="${RM_VAL_FRACTION:-0.1}"

# Policy
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
POLICY_ROOT="${POLICY_ROOT:-rga_checkpoints}"
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
POLICY_CHECKPOINT="$PROJECT_ROOT/${MODEL_SLUG}/${POLICY_ROOT}/${DATASET}/policy_rga.pth"
MODEL_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${CHECKPOINT_ROOT}/${DATASET}"
COLD_START_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${COLD_START_ROOT}/${DATASET}"
D_EFF_DIR="$MODEL_DIR/FinetuneData_${DATASET}"
ARG_MODEL_DIR="${ARG_MODEL_DIR:-$MODEL_DIR}"

# ---- Stage checkpointing ----------------------------------------------------
STATE_DIR="$PROJECT_ROOT/${MODEL_SLUG}/state/rga/${DATASET}"
mkdir -p "$STATE_DIR"
stage_done() { [[ -f "$STATE_DIR/$1.done" ]]; }
mark_done()  { touch "$STATE_DIR/$1.done"; echo "  checkpoint: $STATE_DIR/$1.done"; }

echo "================================================"
echo "  RLHF Pipeline"
echo "  Dataset  : $DATASET (index $DATASET_IDX)"
echo "  Model    : $HF_MODEL  (slug: $MODEL_SLUG)"
echo "  Backend  : $([ "${USE_VLLM_SERVER}" = "1" ] && echo "OpenAI-compatible ($LOCAL_BASE_URL)" || echo "HuggingFace transformers")"
echo "  BoN      : $([ "${BEST_OF_N:-1}" -gt 1 ] && echo "N=$BEST_OF_N temp=$BON_TEMPERATURE" || echo "disabled")"
echo "  Started  : $(date)"
echo "================================================"

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "ERROR: base checkpoint not found: $MODEL_DIR"
    echo "  Run scripts/train.sh first."
    exit 1
fi

# Helper to build --coldstart_dirs flag
_coldstart_dirs_flag() {
    local _dirs="${COLDSTART_DIRS:-}"
    [[ -d "$COLD_START_DIR" ]] && _dirs="${_dirs:+$_dirs }$COLD_START_DIR"
    [[ -d "$D_EFF_DIR"      ]] && _dirs="${_dirs:+$_dirs }$D_EFF_DIR"
    [[ -d "$COLD_START_DIR/rga_rejected" ]] && _dirs="${_dirs:+$_dirs }$COLD_START_DIR/rga_rejected"
    [[ -d "$D_EFF_DIR/rga_rejected"      ]] && _dirs="${_dirs:+$_dirs }$D_EFF_DIR/rga_rejected"
    [[ -n "$_dirs" ]] && echo "--coldstart_dirs $_dirs"
}

# ---- Stage 1a: Generate candidates ------------------------------------------
if stage_done stage1a_gen_candidates || stage_done stage1_collect; then
    echo "[Stage 1a] Gen candidates ... SKIPPED"
else
    echo "[Stage 1a] Gen candidates ..."

    uv run rga \
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

    uv run rga \
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

    uv run rga \
        --dataset "$DATASET" --phase train_rm \
        --preference_dir "$PREFERENCE_DIR" \
        --rm_checkpoint "$RM_CHECKPOINT" \
        --rm_epochs "$RM_EPOCHS" --rm_lr "$RM_LR" \
        --rm_batch_size "$RM_BATCH_SIZE" \
        --rm_hidden_dim "$RM_HIDDEN_DIM" --rm_output_dim "$RM_OUTPUT_DIM" \
        --rm_val_fraction "$RM_VAL_FRACTION" \
        --both_wrong_weight "$BOTH_WRONG_WEIGHT" \
        --both_correct_weight "$BOTH_CORRECT_WEIGHT" \
        --rm_loss "$RM_LOSS"

    mark_done stage2_train_rm
fi

# ---- Stage 3: Train policy --------------------------------------------------
if stage_done stage3_train_policy; then
    echo "[Stage 3] Train policy ... SKIPPED"
else
    echo "[Stage 3] Train policy ..."

    uv run rga \
        --dataset "$DATASET" --phase train_policy \
        --llm_name "$HF_MODEL" --model_dir "$MODEL_DIR" \
        --rm_checkpoint "$RM_CHECKPOINT" \
        --policy_checkpoint "$POLICY_CHECKPOINT" \
        --dataset_json "$DATASET_JSON" \
        --num_tasks "$POLICY_NUM_TASKS" \
        --policy_epochs "$POLICY_EPOCHS" --policy_lr "$POLICY_LR" \
        --kl_coeff "$KL_COEFF" --samples_per_task "$SAMPLES_PER_TASK" \
        --grad_accum_steps "$GRAD_ACCUM_STEPS" --seed "$SEED" \
        --lambda_eff "$LAMBDA_EFF" $SQUASH_FLAG

    mark_done stage3_train_policy
fi

POLICY_DIR="$(dirname "$POLICY_CHECKPOINT")"
MODEL_TYPE="rga"
GRAPHS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${GRAPHS_ROOT}/${MODEL_TYPE}"
GRAPHS_FILE="$GRAPHS_DIR/${DATASET}_graphs.jsonl"
NO_EF_FLAG=""; [[ "$NO_EF" == "1" ]] && NO_EF_FLAG="--no_ef"

# Best-of-N flags
BON_FLAGS=""
if [[ "${BEST_OF_N:-1}" -gt 1 ]]; then
    ROLE_EMB_FILE="$COLD_START_DIR/precomputed_role_embeddings.pkl"
    if [[ -f "$RM_CHECKPOINT" && -f "$ROLE_EMB_FILE" ]]; then
        # V_max = MAX_AGENTS, as in the policy reward.
        BON_FLAGS="--best_of_n $BEST_OF_N --bon_temperature $BON_TEMPERATURE --rm_checkpoint $RM_CHECKPOINT --role_emb_path $ROLE_EMB_FILE --size_lambda $SIZE_LAMBDA --v_max $MAX_AGENTS $SQUASH_FLAG"
        echo "  [BoN] N=$BEST_OF_N  temperature=$BON_TEMPERATURE  size_lambda=$SIZE_LAMBDA  V_max=$MAX_AGENTS"
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

    RESULTS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/rga"
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
echo "  Results: ${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/rga/summary.jsonl"
echo "  Finished: $(date)"
echo "================================================"
