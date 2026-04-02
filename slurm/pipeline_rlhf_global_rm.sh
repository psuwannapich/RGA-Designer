#!/bin/bash
#SBATCH --job-name=arg_rlhf_global_rm
#SBATCH --output=logs/pipeline_rlhf_global_rm_%j.out
#SBATCH --error=logs/pipeline_rlhf_global_rm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH -p gpu
#SBATCH --time=10:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# Global RLHF pipeline — trains ONE reward model on preference data pooled
# across ALL datasets, then fine-tunes a per-dataset policy with it.
#
# Prerequisites: pipeline_rlhf.sh (collect phase) must have completed for
#   every dataset so that preference pair shards already exist.
#
# Stages (sequential):
#   1. train_rm      single global reward model on all preference dirs
#   2. train_policy  per-dataset policy fine-tuning with the global RM
#   3. benchmark     per-dataset evaluation of the global-RM policy
#
# Optional env vars:
#   HF_MODEL           HuggingFace model ID            (default: Qwen/Qwen3-4B)
#   DISABLE_THINKING   1 = no-thinking mode (Qwen3)    (default: 1)
#   CHECKPOINT_ROOT    Phase-1/2 checkpoint sub-dir     (default: checkpoints)
#   PREFERENCE_ROOT    preference data sub-dir           (default: rlhf_data)
#   RM_ROOT            reward model checkpoint sub-dir   (default: rlhf_checkpoints)
#   POLICY_ROOT        policy checkpoint sub-dir         (default: rlhf_checkpoints_global)
#   POLICY_NUM_TASKS   tasks to sample for policy train  (default: 200)
#   EVAL_BATCH         benchmark batch size              (default: 2)
#   RESULTS_ROOT       results sub-dir                  (default: benchmark_results)
#   RUN_NAME           unique name for this run; namespaces policy checkpoints,
#                      graphs, state flags, and global RM so re-runs with
#                      different settings never overwrite each other.
#                      (default: "default")
#                      Example: RUN_NAME=kl01_ep30 sbatch pipeline_rlhf_global_rm.sh
# ---------------------------------------------------------------------------

set -euo pipefail

RUN_NAME=better_split

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
DECISION_METHODS=(FinalRefer FinalRefer FinalRefer FinalRefer FinalWriteCode FinalRefer)
TASK_SPLIT_PATHS=(
    "benchmark_datasets/gsm8k/task_split_gsm8k.json"
    "benchmark_datasets/AQuA/task_split_aqua.json"
    "benchmark_datasets/MultiArith/task_split_multiarith.json"
    "benchmark_datasets/SVAMP/task_split_svamp.json"
    "benchmark_datasets/humaneval/task_split_humaneval.json"
    "benchmark_datasets/MMLU/task_split_mmlu.json"
)

# ---- Configuration ----------------------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"
MODEL_SLUG="${HF_MODEL//\//-}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
MODEL_SLUG="${MODEL_SLUG}-vllm-$([ "${DISABLE_THINKING}" = "1" ] && echo no_thinking || echo thinking)"
export DISABLE_THINKING PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

SEED="${SEED:-42}"

# Run name — used to namespace all mutable outputs so different settings
# can coexist under the same MODEL_SLUG without overwriting each other.
RUN_NAME="${RUN_NAME:-default}"
MODEL_TYPE="rlhf_global_rm_${RUN_NAME}"

# Reward model
PREFERENCE_ROOT="${PREFERENCE_ROOT:-rlhf_data}"
RM_ROOT="${RM_ROOT:-rlhf_checkpoints}"
RM_EPOCHS="${RM_EPOCHS:-20}"
RM_LR="${RM_LR:-1e-4}"
RM_BATCH_SIZE="${RM_BATCH_SIZE:-32}"
RM_HIDDEN_DIM="${RM_HIDDEN_DIM:-256}"
RM_OUTPUT_DIM="${RM_OUTPUT_DIM:-128}"
RM_VAL_FRACTION="${RM_VAL_FRACTION:-0.1}"
BOTH_WRONG_WEIGHT="${BOTH_WRONG_WEIGHT:-0.2}"

# Policy (per-dataset, uses global RM)
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
POLICY_ROOT="${POLICY_ROOT:-rlhf_checkpoints_global}"   # separate from per-dataset policies
POLICY_EPOCHS="${POLICY_EPOCHS:-20}"
POLICY_NUM_TASKS="${POLICY_NUM_TASKS:-200}"
POLICY_LR="${POLICY_LR:-5e-6}"
KL_COEFF="${KL_COEFF:-0.1}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-2}"

# Benchmark
EVAL_BATCH="${EVAL_BATCH:-2}"
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_results}"
GRAPHS_ROOT="${GRAPHS_ROOT:-graphs}"
LIMIT="${LIMIT:-}"
NO_EF="${NO_EF:-0}"

# Global reward model checkpoint (namespaced by RUN_NAME)
GLOBAL_RM_CHECKPOINT="$PROJECT_ROOT/${MODEL_SLUG}/${RM_ROOT}/global_${RUN_NAME}/reward_model.pth"

# ---- vLLM inference server --------------------------------------------------
USE_VLLM_SERVER="${USE_VLLM_SERVER:-1}"
VLLM_PORT="${VLLM_PORT:-6889}"           # offset from per-dataset port range (6789–6836)
VLLM_TP="${VLLM_TP:-2}"
VLLM_SERVE_DIR="${VLLM_SERVE_DIR:-/home/users/psuwannapichat/work_space/vllm_temp}"
VLLM_CHAT_TEMPLATE="${VLLM_CHAT_TEMPLATE:-${VLLM_SERVE_DIR}/qwen3_nonthinking.jinja}"
VLLM_PID=""

# ---- Stage checkpointing (namespaced by RUN_NAME) ---------------------------
STATE_DIR="$PROJECT_ROOT/${MODEL_SLUG}/state/rlhf_global_${RUN_NAME}"
mkdir -p "$STATE_DIR"

stage_done() { [[ -f "$STATE_DIR/$1.done" ]]; }
mark_done()  { touch "$STATE_DIR/$1.done"; echo "  ✔ checkpoint: $STATE_DIR/$1.done"; }

# ---- Header -----------------------------------------------------------------
echo "════════════════════════════════════════════════"
echo "  Global RLHF Pipeline"
echo "  Job              : $SLURM_JOB_ID"
echo "  Node             : $SLURM_NODELIST"
echo "  Model            : $HF_MODEL  (slug: $MODEL_SLUG)"
echo "  Datasets         : ${DATASETS[*]}"
echo "  Run name         : $RUN_NAME"
echo "  Model type       : $MODEL_TYPE"
echo "  Global RM        : $GLOBAL_RM_CHECKPOINT"
echo "  Policy root      : ${MODEL_SLUG}/${POLICY_ROOT}/<dataset>"
echo "  RM epochs        : $RM_EPOCHS"
echo "  Policy epochs    : $POLICY_EPOCHS"
echo "  Started at       : $(date)"
echo "════════════════════════════════════════════════"

# ---- Stage 1: Train global reward model -------------------------------------
echo ""
if stage_done stage1_train_global_rm; then
    echo "▶ [Stage 1] Train global reward model ... SKIPPED (already done)"
else
    echo "▶ [Stage 1] Train global reward model ..."

    # Collect all preference dirs that exist
    PREF_DIRS=()
    for DATASET in "${DATASETS[@]}"; do
        PREF_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${PREFERENCE_ROOT}/${DATASET}"
        if [[ -d "$PREF_DIR" ]] && ls "$PREF_DIR"/*.pkl >/dev/null 2>&1; then
            PREF_DIRS+=("$PREF_DIR")
            echo "  Found preference data: $PREF_DIR"
        else
            echo "  [warn] No preference shards in $PREF_DIR — skipping"
        fi
    done

    if [[ ${#PREF_DIRS[@]} -eq 0 ]]; then
        echo "ERROR: No preference data found for any dataset."
        echo "  Run pipeline_rlhf.sh (collect phase) for each dataset first."
        exit 1
    fi

    echo "  Pooling ${#PREF_DIRS[@]} preference dirs into one global reward model."
    mkdir -p "$(dirname "$GLOBAL_RM_CHECKPOINT")"

    uv run rlhf \
        --dataset        gsm8k \
        --phase          train_rm \
        --preference_dirs "${PREF_DIRS[@]}" \
        --rm_checkpoint  "$GLOBAL_RM_CHECKPOINT" \
        --rm_epochs      "$RM_EPOCHS" \
        --rm_lr          "$RM_LR" \
        --rm_batch_size  "$RM_BATCH_SIZE" \
        --rm_hidden_dim  "$RM_HIDDEN_DIM" \
        --rm_output_dim  "$RM_OUTPUT_DIM" \
        --rm_val_fraction "$RM_VAL_FRACTION" \
        --both_wrong_weight "$BOTH_WRONG_WEIGHT"

    mark_done stage1_train_global_rm
    echo "  ✓ Global reward model → $GLOBAL_RM_CHECKPOINT"
fi

NO_EF_FLAG=""
[[ "$NO_EF" == "1" ]] && NO_EF_FLAG="--no_ef"

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$i]}"
    MAX_AGENTS="${DATASET_MAX_AGENTS[$i]}"
    DECISION="${DECISION_METHODS[$i]}"
    TASK_SPLIT_RAW="${TASK_SPLIT_PATHS[$i]}"

    MODEL_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${CHECKPOINT_ROOT}/${DATASET}"
    POLICY_CHECKPOINT="$PROJECT_ROOT/${MODEL_SLUG}/${POLICY_ROOT}/${RUN_NAME}/${DATASET}/policy_rlhf_global_rm.pth"
    POLICY_DIR="$(dirname "$POLICY_CHECKPOINT")"
    GRAPHS_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${GRAPHS_ROOT}/${MODEL_TYPE}"
    GRAPHS_FILE="$GRAPHS_DIR/${DATASET}_graphs.jsonl"

    echo ""
    echo "════════════════════════════════════════════════"
    echo "  Dataset: $DATASET"
    echo "════════════════════════════════════════════════"

    # -- Stage 2: Train policy ------------------------------------------------
    echo ""
    if stage_done "stage2_policy_${DATASET}"; then
        echo "▶ [Stage 2] Train policy ($DATASET) ... SKIPPED (already done)"
    else
        echo "▶ [Stage 2] Train RLHF policy ($DATASET) with global reward model ..."

        if [[ ! -d "$MODEL_DIR" ]]; then
            echo "  [warn] Base checkpoint not found: $MODEL_DIR — skipping $DATASET"
            continue
        fi

        mkdir -p "$POLICY_DIR"

        uv run rlhf \
            --dataset            "$DATASET" \
            --phase              train_policy \
            --dataset_json       "$DATASET_JSON" \
            --model_dir          "$MODEL_DIR" \
            --rm_checkpoint      "$GLOBAL_RM_CHECKPOINT" \
            --policy_checkpoint  "$POLICY_CHECKPOINT" \
            --num_tasks          "$POLICY_NUM_TASKS" \
            --policy_epochs      "$POLICY_EPOCHS" \
            --policy_lr          "$POLICY_LR" \
            --kl_coeff           "$KL_COEFF" \
            --samples_per_task   "$SAMPLES_PER_TASK" \
            --seed               "$SEED"

        mark_done "stage2_policy_${DATASET}"
        echo "  ✓ Policy → $POLICY_CHECKPOINT"
    fi

    # -- Stage 3a: Generate graphs --------------------------------------------
    echo ""
    if stage_done "stage3a_graphs_${DATASET}"; then
        echo "▶ [Stage 3a] Generate graphs ($DATASET) ... SKIPPED (already done)"
    else
        echo "▶ [Stage 3a] Generate graphs ($DATASET) ..."

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

        mark_done "stage3a_graphs_${DATASET}"
        echo "  ✓ Graphs → $GRAPHS_FILE"
    fi
done

echo ""
echo "════════════════════════════════════════════════"
echo "  Global RLHF pipeline complete"
echo "  Run name   : $RUN_NAME"
echo "  Global RM  : $GLOBAL_RM_CHECKPOINT"
echo "  Graphs     : ${MODEL_SLUG}/${GRAPHS_ROOT}/${MODEL_TYPE}/"
echo "  To benchmark: MODEL_TYPE=$MODEL_TYPE sbatch slurm/benchmark_pregraph.sh"
echo "  Finished   : $(date)"
echo "════════════════════════════════════════════════"
