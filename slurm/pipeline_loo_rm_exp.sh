#!/bin/bash
#SBATCH --job-name=loo_rm
#SBATCH --output=logs/loo_rm_%j.out
#SBATCH --error=logs/loo_rm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# Leave-one-out / leave-group-out reward model pipeline.
#
# Trains ONE reward model PER TARGET DATASET, where each RM is trained on
# preference pairs from ALL OTHER datasets (leaving the target out).  The
# policy for the target dataset is then fine-tuned with that dataset-specific
# RM and benchmarked.  Purpose: measure how well a reward model generalises
# to task types it was never explicitly trained on.
#
# Two exclusion modes (LOO_MODE):
#
#   loo  (default) — Leave-One-Out
#         RM for gsm8k is trained on: aqua multiarith svamp humaneval mmlu
#         RM for humaneval is trained on: gsm8k aqua multiarith svamp mmlu
#         ... etc.
#
#   lgo  — Leave-Group-Out
#         Datasets are grouped by task type:
#           math     : gsm8k aqua multiarith svamp
#           code     : humaneval
#           knowledge: mmlu
#         RM for any math dataset is trained on: humaneval mmlu
#         RM for humaneval is trained on: gsm8k aqua multiarith svamp mmlu
#         RM for mmlu is trained on: gsm8k aqua multiarith svamp humaneval
#
# Prerequisite:
#   pipeline_arg.sh must have completed          → ARG checkpoints
#   pipeline_rlhf_exp.sh stages 1a+1b completed  → preference pairs
#
# Reads from:
#   <llm_slug>/arg/<arg_run>/checkpoints/<dataset>/              (policy init)
#   <llm_slug>/<src_exp_name>/<src_run>/rlhf_data/<dataset>/     (preference pairs)
#
# Writes to:
#   <llm_slug>/<exp_name>/<run>/
#     loo_rm/<dataset>/reward_model.pth    (one RM per target dataset)
#     rlhf_checkpoints/<dataset>/policy_rlhf.pth
#     graphs/rlhf/<dataset>_graphs.jsonl
#     graphs/rlhf_train/<dataset>_graphs.jsonl
#     benchmark_results/pregraph/rlhf/<dataset>.jsonl
#     benchmark_results/pregraph/rlhf_train/<dataset>.jsonl
#     state/
#
# Usage:
#   sbatch slurm/pipeline_loo_rm_exp.sh
#
#   # Leave-group-out mode:
#   LOO_MODE=lgo EXP_NAME=lgo_rm sbatch slurm/pipeline_loo_rm_exp.sh
#
#   # Specific source experiment:
#   SRC_EXP_NAME=kl05 RUN=2 EXP_NAME=loo_rm_kl05 \
#       sbatch slurm/pipeline_loo_rm_exp.sh
#
# Optional env vars:
#   HF_MODEL            HuggingFace model ID                   (default: Qwen/Qwen3-4B)
#   LLM_SLUG            directory name for the LLM             (default: HF_MODEL with / → -)
#   LOO_MODE            loo (leave-one-out) or lgo (leave-group-out) (default: loo)
#   SRC_EXP_NAME        experiment to read preference data from (default: baseline)
#   RUN                 repetition index 0-4                    (default: 0)
#   ARG_RUN             which ARG run to use for policy init    (default: same as RUN)
#   EXP_NAME            name for this experiment                (default: loo_rm)
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

# ---- Leave-group-out groups (LGO mode only) ---------------------------------
# In LGO mode, when the target belongs to a group, all datasets in the SAME
# group are excluded from that target's RM training.
declare -A DATASET_GROUP
DATASET_GROUP[gsm8k]="math"
DATASET_GROUP[aqua]="math"
DATASET_GROUP[multiarith]="math"
DATASET_GROUP[svamp]="math"
DATASET_GROUP[humaneval]="code"
DATASET_GROUP[mmlu]="knowledge"

# ---- Model configuration ----------------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"

if [[ "$HF_MODEL" == *"Llama"* ]] || [[ "$HF_MODEL" == *"llama"* ]]; then
    IS_LLAMA=1
else
    IS_LLAMA=0
fi

DISABLE_THINKING="${DISABLE_THINKING:-$([ "$IS_LLAMA" = "1" ] && echo 0 || echo 1)}"
LLM_SLUG="${LLM_SLUG:-${HF_MODEL//\//-}}"

# Exclusion mode
LOO_MODE="${LOO_MODE:-loo}"   # loo | lgo
if [[ "$LOO_MODE" != "loo" && "$LOO_MODE" != "lgo" ]]; then
    echo "ERROR: LOO_MODE must be 'loo' or 'lgo', got '$LOO_MODE'" >&2
    exit 1
fi

# This experiment
EXP_NAME="${EXP_NAME:-loo_rm}"
RUN="${RUN:-0}"
ARG_RUN="${RUN:-0}"

# Source experiment — where to read already-collected preference pairs from
SRC_EXP_NAME="${SRC_EXP_NAME:-kl05}"
SRC_RUN="${RUN:-0}"

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
BEST_OF_N="${BEST_OF_N:-5}"
BON_TEMPERATURE="${BON_TEMPERATURE:-1}"

# ---- Directory layout -------------------------------------------------------
SRC_BASE="$PROJECT_ROOT/${LLM_SLUG}/${SRC_EXP_NAME}/${SRC_RUN}"
ARG_BASE="$PROJECT_ROOT/${LLM_SLUG}/arg/${ARG_RUN}"
RLHF_BASE="$PROJECT_ROOT/${LLM_SLUG}/${EXP_NAME}/${RUN}"

# ---- vLLM inference server --------------------------------------------------
USE_VLLM_SERVER="${USE_VLLM_SERVER:-1}"
PORT_OFFSET="${PORT_OFFSET:-0}"
VLLM_PORT="${VLLM_PORT:-$((9680 + RUN * 10 + PORT_OFFSET))}"
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
# stage_done / mark_done: global scope (uses STATE_DIR set explicitly by caller).
STATE_DIR="$RLHF_BASE/state"
stage_done() { [[ -f "$STATE_DIR/$1.done" ]]; }
mark_done()  { touch "$STATE_DIR/$1.done"; echo "  ✔ checkpoint: $STATE_DIR/$1.done"; }

# ds_done / ds_mark: per-dataset scope (uses DATASET loop variable directly,
# never relies on the mutable STATE_DIR global).
ds_done() { [[ -f "$RLHF_BASE/state/${DATASET}/$1.done" ]]; }
ds_mark() { local _d="$RLHF_BASE/state/${DATASET}"; mkdir -p "$_d"; touch "$_d/$1.done"; echo "  ✔ checkpoint: $_d/$1.done"; }

# ---- Helper: compute LOO/LGO excluded and included preference dirs ----------
# Sets PREF_DIRS and LOO_EXCLUDED_DISPLAY for the current $DATASET.
_compute_pref_dirs() {
    PREF_DIRS=()
    LOO_EXCLUDED_DISPLAY=()

    local target_group=""
    if [[ "$LOO_MODE" == "lgo" ]]; then
        target_group="${DATASET_GROUP[$DATASET]}"
    fi

    for ds in "${DATASETS[@]}"; do
        local pref_dir="$SRC_BASE/rlhf_data/${ds}"
        local exclude=0

        if [[ "$LOO_MODE" == "lgo" ]]; then
            [[ "${DATASET_GROUP[$ds]}" == "$target_group" ]] && exclude=1
        else
            [[ "$ds" == "$DATASET" ]] && exclude=1
        fi

        if [[ "$exclude" == "1" ]]; then
            LOO_EXCLUDED_DISPLAY+=("$ds")
        elif [[ -d "$pref_dir" ]]; then
            PREF_DIRS+=("$pref_dir")
        else
            echo "  WARNING: no preference data for $ds at $pref_dir — skipping from RM training for $DATASET"
        fi
    done
}

# ---- Sanity checks ----------------------------------------------------------
if [[ ! -f "$ARG_BASE/checkpoints/${DATASETS[0]}/ef_best_model.pth" ]]; then
    echo "ERROR: ARG checkpoint not found under $ARG_BASE/checkpoints/"
    echo "  Run pipeline_arg.sh first (LLM_SLUG=$LLM_SLUG, ARG_RUN=$ARG_RUN)."
    exit 1
fi

# Verify at least one dataset has preference data
_any_pref_found=0
for _ds in "${DATASETS[@]}"; do
    [[ -d "$SRC_BASE/rlhf_data/${_ds}" ]] && { _any_pref_found=1; break; }
done
if [[ "$_any_pref_found" == "0" ]]; then
    echo "ERROR: no preference data found in $SRC_BASE/rlhf_data/." >&2
    echo "  Run pipeline_rlhf_exp.sh stages 1a+1b first (SRC_EXP_NAME=$SRC_EXP_NAME, SRC_RUN=$SRC_RUN)." >&2
    exit 1
fi

mkdir -p "$RLHF_BASE/state"

# ---- Header -----------------------------------------------------------------
echo "════════════════════════════════════════════════"
echo "  Leave-$([ "$LOO_MODE" = "lgo" ] && echo "Group" || echo "One")-Out Reward Model Pipeline"
echo "  Job ${SLURM_JOB_ID:-local}"
echo "  Node              : ${SLURM_NODELIST:-local}"
echo "  Model             : $HF_MODEL"
echo "  LLM slug          : $LLM_SLUG"
echo "  Exclusion mode    : $LOO_MODE"
echo "  Experiment        : $EXP_NAME  (run $RUN)"
echo "  Source exp        : ${LLM_SLUG}/${SRC_EXP_NAME}/${SRC_RUN}/"
echo "  ARG base          : ${LLM_SLUG}/arg/${ARG_RUN}/"
echo "  Output base       : ${LLM_SLUG}/${EXP_NAME}/${RUN}/"
echo "  RM epochs         : $RM_EPOCHS  lr=$RM_LR"
echo "  Policy epochs     : $POLICY_EPOCHS  lr=$POLICY_LR  kl=$KL_COEFF  lambda_eff=$LAMBDA_EFF"
echo "  Best-of-N         : $([ "${BEST_OF_N:-1}" -gt 1 ] && echo "N=$BEST_OF_N  temp=$BON_TEMPERATURE" || echo "disabled")"
echo "  vLLM server       : $([ "$USE_VLLM_SERVER" = "1" ] && echo "enabled (port $VLLM_PORT, tp=$VLLM_TP)" || echo "disabled (HF backend)")"
if [[ "$LOO_MODE" == "lgo" ]]; then
    echo "  LGO groups        : math={gsm8k,aqua,multiarith,svamp}  code={humaneval}  knowledge={mmlu}"
fi
echo "  Started at        : $(date)"
echo "════════════════════════════════════════════════"

if [[ "$USE_VLLM_SERVER" == "1" ]]; then
    export LOCAL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export LOCAL_API_KEY="EMPTY"
    export USE_VLLM_SERVER USE_VLLM=0
fi

# ============================================================================
# Pass 1 (no vLLM): per-dataset LOO RM training + policy + graph generation
#
# For each target dataset:
#   Stage 1 : train LOO/LGO reward model (excludes target or its group)
#   Stage 2 : fine-tune policy with that RM
#   Stage 3a: generate test-set graphs
#   Stage 4a: generate train-set graphs (overfitting check)
# ============================================================================
NO_EF_FLAG=""
[[ "$NO_EF" == "1" ]] && NO_EF_FLAG="--no_ef"

echo ""
echo "════════════ Pass 1: LOO RM Training + Policy + Graph Generation (all datasets) ════════════"
_ensure_vllm_stopped

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$i]}"
    TASK_SPLIT_RAW="${TASK_SPLIT_PATHS[$i]}"

    ARG_CHECKPOINT_DIR="$ARG_BASE/checkpoints/${DATASET}"
    ARG_COLD_START_DIR="$ARG_BASE/ColdStartData/${DATASET}"
    LOO_RM_CHECKPOINT="$RLHF_BASE/loo_rm/${DATASET}/reward_model.pth"
    POLICY_CHECKPOINT="$RLHF_BASE/rlhf_checkpoints/${DATASET}/policy_rlhf.pth"
    POLICY_DIR="$(dirname "$POLICY_CHECKPOINT")"
    GRAPHS_DIR="$RLHF_BASE/graphs/rlhf"
    GRAPHS_FILE="$GRAPHS_DIR/${DATASET}_graphs.jsonl"
    TRAIN_GRAPHS_DIR="$RLHF_BASE/graphs/rlhf_train"
    TRAIN_GRAPHS_FILE="$TRAIN_GRAPHS_DIR/${DATASET}_graphs.jsonl"

    mkdir -p "$RLHF_BASE/state/${DATASET}"

    # Compute which preference dirs to include for this target
    _compute_pref_dirs

    echo ""
    echo "════════════ $DATASET ════════════"
    echo "  LOO excluded : ${LOO_EXCLUDED_DISPLAY[*]}"
    echo "  RM trained on: $(printf '%s ' "${PREF_DIRS[@]}" | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"

    if [[ ! -f "$ARG_CHECKPOINT_DIR/ef_best_model.pth" ]]; then
        echo "  WARNING: ARG checkpoint missing — skipping $DATASET"
        continue
    fi

    if [[ ${#PREF_DIRS[@]} -eq 0 ]]; then
        echo "  WARNING: no preference data available for RM training — skipping $DATASET"
        continue
    fi

    # ---- Stage 1: Train LOO reward model for this target --------------------
    if ds_done stage1_train_loo_rm; then
        echo "▶ [Stage 1] Train LOO RM ... SKIPPED (already done)"
    else
        echo "▶ [Stage 1] Training LOO RM for $DATASET (${#PREF_DIRS[@]} source dataset(s)) ..."
        for _d in "${PREF_DIRS[@]}"; do echo "  + $_d"; done
        mkdir -p "$(dirname "$LOO_RM_CHECKPOINT")"

        uv run rlhf \
            --dataset             "$DATASET" \
            --phase               train_rm \
            --preference_dirs     "${PREF_DIRS[@]}" \
            --rm_checkpoint       "$LOO_RM_CHECKPOINT" \
            --rm_epochs           "$RM_EPOCHS" \
            --rm_lr               "$RM_LR" \
            --rm_batch_size       "$RM_BATCH_SIZE" \
            --rm_hidden_dim       "$RM_HIDDEN_DIM" \
            --rm_output_dim       "$RM_OUTPUT_DIM" \
            --rm_val_fraction     "$RM_VAL_FRACTION" \
            --both_wrong_weight   "$BOTH_WRONG_WEIGHT" \
            --both_correct_weight "$BOTH_CORRECT_WEIGHT" \
            --rm_loss             "$RM_LOSS"

        ds_mark stage1_train_loo_rm
        echo "  ✓ LOO RM → $LOO_RM_CHECKPOINT"
    fi

    # ---- Stage 2: Train policy with this dataset's LOO RM -------------------
    if ds_done stage2_train_policy; then
        echo "▶ [Stage 2] Train policy ... SKIPPED (already done)"
    else
        echo "▶ [Stage 2] Train policy ($DATASET) using LOO RM ..."

        uv run rlhf \
            --dataset            "$DATASET" \
            --phase              train_policy \
            --llm_name           "$HF_MODEL" \
            --model_dir          "$ARG_CHECKPOINT_DIR" \
            --rm_checkpoint      "$LOO_RM_CHECKPOINT" \
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

        ds_mark stage2_train_policy
        echo "  ✓ Policy → $POLICY_CHECKPOINT"
    fi

    # Compute BoN flags once for both graph generation stages
    BON_FLAGS=""
    if [[ "${BEST_OF_N:-1}" -gt 1 ]]; then
        ROLE_EMB_FILE="$ARG_COLD_START_DIR/precomputed_role_embeddings.pkl"
        if [[ -f "$LOO_RM_CHECKPOINT" && -f "$ROLE_EMB_FILE" ]]; then
            BON_FLAGS="--best_of_n $BEST_OF_N --bon_temperature $BON_TEMPERATURE --rm_checkpoint $LOO_RM_CHECKPOINT --role_emb_path $ROLE_EMB_FILE"
        else
            echo "  [BoN] Skipped (missing RM checkpoint or role embeddings for $DATASET)"
        fi
    fi

    # ---- Stage 3a: Generate test-set graphs ---------------------------------
    if ds_done stage3a_gen_graphs; then
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

        ds_mark stage3a_gen_graphs
        echo "  ✓ Graphs → $GRAPHS_FILE"
    fi

    # ---- Stage 4a: Generate train-set graphs (overfitting check) ------------
    if ds_done stage4a_gen_graphs_train; then
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

        ds_mark stage4a_gen_graphs_train
        echo "  ✓ Train graphs → $TRAIN_GRAPHS_FILE"
    fi
done

# ============================================================================
# Pass 2 (vLLM on): all benchmarks — test set then train set
# ============================================================================
echo ""
echo "════════════ Pass 2: Benchmark (all datasets) ════════════"
_ensure_vllm_running

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    DECISION="${DECISION_METHODS[$i]}"

    GRAPHS_FILE="$RLHF_BASE/graphs/rlhf/${DATASET}_graphs.jsonl"
    RESULTS_DIR="$RLHF_BASE/benchmark_results/pregraph/rlhf"
    TRAIN_GRAPHS_FILE="$RLHF_BASE/graphs/rlhf_train/${DATASET}_graphs.jsonl"
    TRAIN_RESULTS_DIR="$RLHF_BASE/benchmark_results/pregraph/rlhf_train"

    # Show which datasets were excluded from this target's RM (for context)
    _compute_pref_dirs
    echo ""
    echo "  ── $DATASET  [LOO excluded: ${LOO_EXCLUDED_DISPLAY[*]}]"

    # ---- Stage 3b: Benchmark test set ----------------------------------------
    if ds_done stage3b_benchmark; then
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

        ds_mark stage3b_benchmark
        echo "  ✓ Results → $OUTPUT_FILE"
    fi

    # ---- Stage 4b: Benchmark train set ---------------------------------------
    if ds_done stage4b_benchmark_train; then
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

        ds_mark stage4b_benchmark_train
        echo "  ✓ Train results → $TRAIN_OUTPUT_FILE"
    fi
done

echo ""
echo "════════════════════════════════════════════════"
echo "  LOO RM pipeline complete"
echo "  Mode            : $LOO_MODE"
echo "  Experiment      : ${LLM_SLUG}/${EXP_NAME}/${RUN}/"
echo "  LOO RMs         : ${LLM_SLUG}/${EXP_NAME}/${RUN}/loo_rm/<dataset>/reward_model.pth"
echo "  Test  results   : ${LLM_SLUG}/${EXP_NAME}/${RUN}/benchmark_results/pregraph/rlhf/summary.jsonl"
echo "  Train results   : ${LLM_SLUG}/${EXP_NAME}/${RUN}/benchmark_results/pregraph/rlhf_train/summary.jsonl"
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════"
