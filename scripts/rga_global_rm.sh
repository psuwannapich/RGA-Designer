#!/bin/bash
# ---------------------------------------------------------------------------
# Global RLHF pipeline — trains ONE reward model on preference data pooled
# across ALL datasets, then fine-tunes a per-dataset policy with it.
#
# Prerequisites: scripts/rga.sh (collect phase, stages 1a+1b) must have
#   completed for every dataset so that preference pair shards exist.
#
# Usage:
#   bash scripts/rga_global_rm.sh
#   RUN_NAME=kl02_bon5 bash scripts/rga_global_rm.sh
#
# Key environment variables (all optional):
#   HF_MODEL         HuggingFace model ID or name passed to --llm_name
#                    (default: Qwen/Qwen3-4B)
#   LOCAL_BASE_URL   OpenAI-compatible API base URL.
#                    Set this to use a vLLM server, Ollama, or a commercial API.
#                    Leave unset to use the HuggingFace transformers backend.
#   LOCAL_API_KEY    API key for the above endpoint (default: EMPTY)
#   RUN_NAME         unique label for this run; namespaces all outputs
#                    so different settings can coexist (default: default)
#   KL_COEFF         KL penalty coefficient (default: 0.2)
#   BEST_OF_N        BoN graph selection at inference (default: 5)
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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

# Agent harness: fixed referee, few-shot demos as turns, reply cap.
export RGA_DECISION_METHOD="${RGA_DECISION_METHOD:-FinalReferTurns}"
export RGA_SOLVER_FEWSHOT="${RGA_SOLVER_FEWSHOT:-turns}"
export MAX_AGENT_TOKENS="${MAX_AGENT_TOKENS:-2048}"

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

# ---- Run name (namespaces all mutable outputs) --------------------------------
RUN_NAME="${RUN_NAME:-default}"
MODEL_TYPE="rga_global_rm_${RUN_NAME}"

# ---- Hyperparameters --------------------------------------------------------
SEED="${SEED:-42}"

# Reward model
PREFERENCE_ROOT="${PREFERENCE_ROOT:-rga_data}"
RM_ROOT="${RM_ROOT:-rga_checkpoints}"
RM_EPOCHS="${RM_EPOCHS:-20}"
RM_LR="${RM_LR:-1e-4}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-32}"
RM_HIDDEN_DIM="${RM_HIDDEN_DIM:-256}"
RM_OUTPUT_DIM="${RM_OUTPUT_DIM:-128}"
RM_VAL_FRACTION="${RM_VAL_FRACTION:-0.1}"
BOTH_WRONG_WEIGHT="${BOTH_WRONG_WEIGHT:-0}"
BOTH_CORRECT_WEIGHT="${BOTH_CORRECT_WEIGHT:-0.1}"

# Policy
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
POLICY_ROOT="${POLICY_ROOT:-rga_checkpoints_global}"
POLICY_EPOCHS="${POLICY_EPOCHS:-20}"
POLICY_NUM_TASKS="${POLICY_NUM_TASKS:-200}"
POLICY_LR="${POLICY_LR:-5e-6}"
KL_COEFF="${KL_COEFF:-0.2}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-4}"

# Benchmark / graph generation
EVAL_BATCH="${EVAL_BATCH:-8}"
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_results}"
GRAPHS_ROOT="${GRAPHS_ROOT:-graphs}"
LIMIT="${LIMIT:-}"
NO_EF="${NO_EF:-0}"
BEST_OF_N="${BEST_OF_N:-5}"
BON_TEMPERATURE="${BON_TEMPERATURE:-1}"
COLD_START_ROOT="${COLD_START_ROOT:-ColdStartData}"

GLOBAL_RM_CHECKPOINT="$PROJECT_ROOT/${MODEL_SLUG}/${RM_ROOT}/global_${RUN_NAME}/reward_model.pth"

# ---- Stage checkpointing ----------------------------------------------------
STATE_DIR="$PROJECT_ROOT/${MODEL_SLUG}/state/rga_global_${RUN_NAME}"
mkdir -p "$STATE_DIR"
stage_done() { [[ -f "$STATE_DIR/$1.done" ]]; }
mark_done()  { touch "$STATE_DIR/$1.done"; echo "  checkpoint: $STATE_DIR/$1.done"; }

echo "================================================"
echo "  Global RLHF Pipeline"
echo "  Model    : $HF_MODEL  (slug: $MODEL_SLUG)"
echo "  Backend  : $([ "${USE_VLLM_SERVER}" = "1" ] && echo "OpenAI-compatible ($LOCAL_BASE_URL)" || echo "HuggingFace transformers")"
echo "  Run name : $RUN_NAME"
echo "  Global RM: $GLOBAL_RM_CHECKPOINT"
echo "  BoN      : $([ "${BEST_OF_N:-1}" -gt 1 ] && echo "N=$BEST_OF_N temp=$BON_TEMPERATURE" || echo "disabled")"
echo "  Started  : $(date)"
echo "================================================"

# ---- Stage 1: Train global reward model -------------------------------------
if stage_done stage1_train_global_rm; then
    echo "[Stage 1] Train global reward model ... SKIPPED"
else
    echo "[Stage 1] Train global reward model ..."

    PREF_DIRS=()
    for DS in "${DATASETS[@]}"; do
        PREF_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${PREFERENCE_ROOT}/${DS}"
        if [[ -d "$PREF_DIR" ]] && ls "$PREF_DIR"/*.pkl >/dev/null 2>&1; then
            PREF_DIRS+=("$PREF_DIR")
            echo "  Found preference data: $PREF_DIR"
        else
            echo "  [warn] No preference shards in $PREF_DIR — skipping"
        fi
    done

    if [[ ${#PREF_DIRS[@]} -eq 0 ]]; then
        echo "ERROR: No preference data found. Run scripts/rga.sh (stages 1a+1b) for each dataset first."
        exit 1
    fi

    mkdir -p "$(dirname "$GLOBAL_RM_CHECKPOINT")"

    uv run rga \
        --dataset gsm8k --phase train_rm \
        --preference_dirs "${PREF_DIRS[@]}" \
        --rm_checkpoint "$GLOBAL_RM_CHECKPOINT" \
        --rm_epochs "$RM_EPOCHS" --rm_lr "$RM_LR" \
        --rm_batch_size "$RM_BATCH_SIZE" \
        --rm_hidden_dim "$RM_HIDDEN_DIM" --rm_output_dim "$RM_OUTPUT_DIM" \
        --rm_val_fraction "$RM_VAL_FRACTION" \
        --both_wrong_weight "$BOTH_WRONG_WEIGHT" \
        --both_correct_weight "$BOTH_CORRECT_WEIGHT"

    mark_done stage1_train_global_rm
    echo "  Global RM → $GLOBAL_RM_CHECKPOINT"
fi

NO_EF_FLAG=""; [[ "$NO_EF" == "1" ]] && NO_EF_FLAG="--no_ef"

# ---- Per-dataset policy + graph generation ----------------------------------
for i in "${!DATASETS[@]}"; do
    DS="${DATASETS[$i]}"
    DS_JSON="$PROJECT_ROOT/${DATASET_JSONS[$i]}"
    TASK_SPLIT_RAW="${TASK_SPLIT_PATHS[$i]}"
    DECISION="${DECISION_METHODS[$i]}"
    if [[ "$DECISION" != "FinalWriteCode" ]]; then DECISION="$RGA_DECISION_METHOD"; fi

    MODEL_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${CHECKPOINT_ROOT}/${DS}"
    POLICY_CHECKPOINT="$PROJECT_ROOT/${MODEL_SLUG}/${POLICY_ROOT}/${RUN_NAME}/${DS}/policy_rga_global_rm.pth"
    POLICY_DIR="$(dirname "$POLICY_CHECKPOINT")"
    GRAPHS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${GRAPHS_ROOT}/${MODEL_TYPE}"
    GRAPHS_FILE="$GRAPHS_DIR/${DS}_graphs.jsonl"
    COLD_START_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${COLD_START_ROOT}/${DS}"

    echo ""
    echo "---- Dataset: $DS ----"

    # Stage 2: Train policy
    if stage_done "stage2_policy_${DS}"; then
        echo "[Stage 2] Train policy ($DS) ... SKIPPED"
    else
        echo "[Stage 2] Train policy ($DS) ..."
        if [[ ! -d "$MODEL_DIR" ]]; then
            echo "  [warn] Base checkpoint not found: $MODEL_DIR — skipping"
            continue
        fi
        mkdir -p "$POLICY_DIR"

        uv run rga \
            --dataset "$DS" --phase train_policy \
            --dataset_json "$DS_JSON" \
            --model_dir "$MODEL_DIR" \
            --rm_checkpoint "$GLOBAL_RM_CHECKPOINT" \
            --policy_checkpoint "$POLICY_CHECKPOINT" \
            --num_tasks "$POLICY_NUM_TASKS" \
            --policy_epochs "$POLICY_EPOCHS" --policy_lr "$POLICY_LR" \
            --kl_coeff "$KL_COEFF" --samples_per_task "$SAMPLES_PER_TASK" \
            --seed "$SEED"

        mark_done "stage2_policy_${DS}"
    fi

    # Best-of-N flags
    BON_FLAGS=""
    if [[ "${BEST_OF_N:-1}" -gt 1 ]]; then
        ROLE_EMB_FILE="$COLD_START_DIR/precomputed_role_embeddings.pkl"
        if [[ -f "$GLOBAL_RM_CHECKPOINT" && -f "$ROLE_EMB_FILE" ]]; then
            BON_FLAGS="--best_of_n $BEST_OF_N --bon_temperature $BON_TEMPERATURE --rm_checkpoint $GLOBAL_RM_CHECKPOINT --role_emb_path $ROLE_EMB_FILE"
        else
            echo "  [BoN] WARNING: RM or role embeddings missing — BoN disabled for $DS"
        fi
    fi

    # Stage 3a: Generate graphs
    if stage_done "stage3a_graphs_${DS}"; then
        echo "[Stage 3a] Generate graphs ($DS) ... SKIPPED"
    else
        echo "[Stage 3a] Generate graphs ($DS) ..."
        mkdir -p "$GRAPHS_DIR"

        uv run python experiment/generate_graphs.py \
            --model_path "$POLICY_DIR" --dataset "$DS" \
            --dataset_path "$DS_JSON" --output_file "$GRAPHS_FILE" \
            --model_type "$MODEL_TYPE" \
            ${TASK_SPLIT_RAW:+--task_split_path "$PROJECT_ROOT/$TASK_SPLIT_RAW"} \
            ${LIMIT:+--limit "$LIMIT"} $NO_EF_FLAG $BON_FLAGS

        mark_done "stage3a_graphs_${DS}"
    fi

    # Stage 3b: Benchmark
    if stage_done "stage3b_benchmark_${DS}"; then
        echo "[Stage 3b] Benchmark ($DS) ... SKIPPED"
    else
        echo "[Stage 3b] Benchmark ($DS) ..."

        RESULTS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${RESULTS_ROOT}/pregraph/${MODEL_TYPE}"
        mkdir -p "$RESULTS_DIR"

        uv run python experiment/benchmark_pregraph.py \
            --graphs_file "$GRAPHS_FILE" --dataset "$DS" \
            --llm_name "$HF_MODEL" --decision_method "$DECISION" \
            --output_file "$RESULTS_DIR/${DS}.jsonl" \
            --summary_log_file "$RESULTS_DIR/summary.jsonl" \
            --eval_batch_size "$EVAL_BATCH" --model_type "$MODEL_TYPE" \
            ${LIMIT:+--limit "$LIMIT"}

        mark_done "stage3b_benchmark_${DS}"
    fi
done

echo ""
echo "================================================"
echo "  Global RLHF pipeline complete"
echo "  Run name : $RUN_NAME  |  Model type : $MODEL_TYPE"
echo "  Global RM: $GLOBAL_RM_CHECKPOINT"
echo "  Finished : $(date)"
echo "================================================"
