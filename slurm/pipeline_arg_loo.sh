#!/bin/bash
#SBATCH --job-name=arg_loo
#SBATCH --output=logs/arg_loo_%A_%a.out
#SBATCH --error=logs/arg_loo_%A_%a.err
#SBATCH --array=0-4          # run index 0-4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --time=24:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# ARGDesigner Leave-One-Out / Leave-Group-Out training pipeline.
#
# Trains an ARGDesigner model for each target dataset using ONLY cold-start and
# D_eff artifacts from the OTHER datasets — no LLM re-inference required.
# All source artifacts are read from an existing ARG run.
#
# Two exclusion modes (LOO_MODE):
#
#   loo  (default) — Leave-One-Out
#         Model for gsm8k trained on: aqua multiarith svamp humaneval mmlu
#         Model for aqua trained on: gsm8k multiarith svamp humaneval mmlu
#         ... etc.
#
#   lgo  — Leave-Group-Out
#         Groups: math={gsm8k,aqua,multiarith,svamp}  code={humaneval}  knowledge={mmlu}
#         Model for any math dataset trained on: humaneval mmlu
#         Model for humaneval trained on: gsm8k aqua multiarith svamp mmlu
#         Model for mmlu trained on: gsm8k aqua multiarith svamp humaneval
#
# Stages per target dataset (no vLLM needed for training):
#   Stage 1 : pool cold-start .pt files from source datasets → pretrain
#   Stage 2 : pool FinetuneData .pt files from source datasets → finetune
#   Stage 3a: generate test-set graphs (no vLLM)
#   Stage 4a: generate train-set graphs (no vLLM)
#   Stage 3b: benchmark test set  (vLLM required)
#   Stage 4b: benchmark train set (vLLM required)
#
# Prerequisite:
#   pipeline_arg.sh must have completed for the source ARG run.
#   All source datasets' ColdStartData/ and checkpoints/*/FinetuneData_*/
#   directories must exist under <llm_slug>/arg/<arg_run>/.
#
# Reads artifacts from:
#   <llm_slug>/arg/<arg_run>/ColdStartData/<src_dataset>/        (Stage 1)
#   <llm_slug>/arg/<arg_run>/checkpoints/<src_dataset>/FinetuneData_<src_dataset>/  (Stage 2)
#
# Writes to:
#   <llm_slug>/<exp_name>/<run>/
#     checkpoints/<target_dataset>/best_model.pth
#     checkpoints/<target_dataset>/ef_best_model.pth
#     graphs/arg_designer/<target_dataset>_graphs.jsonl
#     graphs/arg_designer_train/<target_dataset>_graphs.jsonl
#     benchmark_results/pregraph/arg_designer/<target_dataset>.jsonl
#     benchmark_results/pregraph/arg_designer_train/<target_dataset>.jsonl
#     state/<target_dataset>/
#
# Usage:
#   sbatch slurm/pipeline_arg_loo.sh
#
#   # Leave-group-out mode:
#   LOO_MODE=lgo EXP_NAME=arg_lgo sbatch slurm/pipeline_arg_loo.sh
#
#   # Use a specific existing ARG run as source:
#   ARG_RUN=2 RUN=2 EXP_NAME=arg_loo sbatch slurm/pipeline_arg_loo.sh
#
# Optional env vars:
#   HF_MODEL          HuggingFace model ID                   (default: Qwen/Qwen3-4B)
#   LLM_SLUG          directory name for the LLM             (default: HF_MODEL with / → -)
#   LOO_MODE          loo or lgo                              (default: loo)
#   EXP_NAME          name for this experiment                (default: arg_loo_<mode>)
#   RUN               repetition index 0-4                    (default: 0)
#   ARG_RUN           source ARG run index                    (default: same as RUN)
#   PORT_OFFSET       add to vLLM port to avoid collisions    (default: 0)
#   DISABLE_THINKING  1 = no-thinking mode (Qwen3)            (default: 1)
#   PRETRAIN_EPOCHS   pretrain epochs                         (default: 30)
#   PRETRAIN_LR       pretrain learning rate                  (default: 1e-4)
#   FINETUNE_EPOCHS   finetune epochs                         (default: 30)
#   FINETUNE_LR       finetune learning rate                  (default: 5e-5)
#   BATCH_SIZE        training batch size                     (default: 32)
#   EVAL_BATCH        benchmark batch size                    (default: 8)
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
DECISION_METHODS=(FinalRefer FinalRefer FinalRefer FinalRefer FinalWriteCode FinalRefer)

# ---- Leave-group-out groups -------------------------------------------------
declare -A DATASET_GROUP
DATASET_GROUP[gsm8k]="math"
DATASET_GROUP[aqua]="math"
DATASET_GROUP[multiarith]="math"
DATASET_GROUP[svamp]="math"
DATASET_GROUP[humaneval]="code"
DATASET_GROUP[mmlu]="knowledge"

# ---- Model / experiment configuration ---------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"

if [[ "$HF_MODEL" == *"Llama"* ]] || [[ "$HF_MODEL" == *"llama"* ]]; then
    IS_LLAMA=1
else
    IS_LLAMA=0
fi

DISABLE_THINKING="${DISABLE_THINKING:-$([ "$IS_LLAMA" = "1" ] && echo 0 || echo 1)}"
LLM_SLUG="${LLM_SLUG:-${HF_MODEL//\//-}}"

LOO_MODE="${LOO_MODE:-loo}"
if [[ "$LOO_MODE" != "loo" && "$LOO_MODE" != "lgo" ]]; then
    echo "ERROR: LOO_MODE must be 'loo' or 'lgo', got '$LOO_MODE'" >&2
    exit 1
fi

RUN="${RUN:-${SLURM_ARRAY_TASK_ID:-0}}"
ARG_RUN="${ARG_RUN:-$RUN}"
EXP_NAME="${EXP_NAME:-arg_${LOO_MODE}}"

export DISABLE_THINKING LLM_SLUG EXP_NAME PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---- Hyperparameters --------------------------------------------------------
SEED="${SEED:-$((42 + RUN))}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-30}"
PRETRAIN_LR="${PRETRAIN_LR:-1e-4}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-30}"
FINETUNE_LR="${FINETUNE_LR:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
VAL_RATIO="${VAL_RATIO:-0.1}"
SAMPLE_SIZE="${SAMPLE_SIZE:-0}"

EVAL_BATCH="${EVAL_BATCH:-8}"
LIMIT="${LIMIT:-}"
NO_EF="${NO_EF:-0}"

# ---- Directory layout -------------------------------------------------------
ARG_BASE="$PROJECT_ROOT/${LLM_SLUG}/arg/${ARG_RUN}"
EXP_BASE="$PROJECT_ROOT/${LLM_SLUG}/${EXP_NAME}/${RUN}"

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
ds_done() { [[ -f "$EXP_BASE/state/${DATASET}/$1.done" ]]; }
ds_mark() { local _d="$EXP_BASE/state/${DATASET}"; mkdir -p "$_d"; touch "$_d/$1.done"; echo "  ✔ checkpoint: $_d/$1.done"; }

# ---- Helper: compute LOO/LGO source dataset lists ---------------------------
# Sets SRC_DATASETS, SRC_COLD_DIRS, SRC_FINETUNE_DIRS, EXCLUDED_DISPLAY
_compute_source_datasets() {
    SRC_DATASETS=()
    SRC_COLD_DIRS=()
    SRC_FINETUNE_DIRS=()
    EXCLUDED_DISPLAY=()

    local target_group=""
    if [[ "$LOO_MODE" == "lgo" ]]; then
        target_group="${DATASET_GROUP[$DATASET]}"
    fi

    for ds in "${DATASETS[@]}"; do
        local exclude=0
        if [[ "$LOO_MODE" == "lgo" ]]; then
            [[ "${DATASET_GROUP[$ds]}" == "$target_group" ]] && exclude=1
        else
            [[ "$ds" == "$DATASET" ]] && exclude=1
        fi

        if [[ "$exclude" == "1" ]]; then
            EXCLUDED_DISPLAY+=("$ds")
            continue
        fi

        local cold_dir="$ARG_BASE/ColdStartData/${ds}"
        local ft_dir="$ARG_BASE/checkpoints/${ds}/FinetuneData_${ds}"

        if [[ ! -d "$cold_dir" ]]; then
            echo "  WARNING: ColdStartData missing for $ds at $cold_dir — skipping"
            continue
        fi
        if [[ ! -d "$ft_dir" ]]; then
            echo "  WARNING: FinetuneData missing for $ds at $ft_dir — skipping"
            continue
        fi

        SRC_DATASETS+=("$ds")
        SRC_COLD_DIRS+=("$cold_dir")
        SRC_FINETUNE_DIRS+=("$ft_dir")
    done
}

# ---- Sanity checks ----------------------------------------------------------
if [[ ! -d "$ARG_BASE/ColdStartData" ]]; then
    echo "ERROR: ARG artifacts not found at $ARG_BASE"
    echo "  Run pipeline_arg.sh first (LLM_SLUG=$LLM_SLUG, ARG_RUN=$ARG_RUN)."
    exit 1
fi

mkdir -p "$EXP_BASE/state"

# ---- Header -----------------------------------------------------------------
echo "════════════════════════════════════════════════"
echo "  ARGDesigner Leave-$([ "$LOO_MODE" = "lgo" ] && echo "Group" || echo "One")-Out Training Pipeline"
echo "  Job ${SLURM_JOB_ID:-local}"
echo "  Node              : ${SLURM_NODELIST:-local}"
echo "  Model             : $HF_MODEL"
echo "  LLM slug          : $LLM_SLUG"
echo "  Exclusion mode    : $LOO_MODE"
echo "  Experiment        : $EXP_NAME  (run $RUN)"
echo "  Source ARG run    : ${LLM_SLUG}/arg/${ARG_RUN}/"
echo "  Output base       : ${LLM_SLUG}/${EXP_NAME}/${RUN}/"
echo "  Pretrain epochs   : $PRETRAIN_EPOCHS  lr=$PRETRAIN_LR"
echo "  Finetune epochs   : $FINETUNE_EPOCHS  lr=$FINETUNE_LR"
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

NO_EF_FLAG=""
[[ "$NO_EF" == "1" ]] && NO_EF_FLAG="--no_ef"

# ============================================================================
# Pass 1 (no vLLM): per-target pretrain → finetune → graph generation
# ============================================================================
echo ""
echo "════════════ Pass 1: Pretrain + Finetune + Graph Generation (no vLLM) ════════════"
_ensure_vllm_stopped

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$i]}"
    TASK_SPLIT_RAW="${TASK_SPLIT_PATHS[$i]}"

    CKPT_DIR="$EXP_BASE/checkpoints/${DATASET}"
    GRAPHS_DIR="$EXP_BASE/graphs/arg_designer"
    GRAPHS_FILE="$GRAPHS_DIR/${DATASET}_graphs.jsonl"
    TRAIN_GRAPHS_DIR="$EXP_BASE/graphs/arg_designer_train"
    TRAIN_GRAPHS_FILE="$TRAIN_GRAPHS_DIR/${DATASET}_graphs.jsonl"

    mkdir -p "$EXP_BASE/state/${DATASET}" "$CKPT_DIR"

    _compute_source_datasets

    echo ""
    echo "════════════ Target: $DATASET ════════════"
    echo "  Excluded (${LOO_MODE^^}) : ${EXCLUDED_DISPLAY[*]}"
    echo "  Source datasets   : ${SRC_DATASETS[*]}"

    if [[ ${#SRC_DATASETS[@]} -eq 0 ]]; then
        echo "  WARNING: no source datasets available — skipping $DATASET"
        continue
    fi

    # ---- Stage 1: Pretrain on pooled cold-start data -------------------------
    if ds_done stage1_pretrain; then
        echo "▶ [Stage 1] Pretrain ... SKIPPED (already done)"
    else
        echo "▶ [Stage 1] Pretrain on cold-start data from: ${SRC_DATASETS[*]} ..."

        uv run python experiment/pretrain_multi.py \
            --phase          pretrain \
            --src_datasets   "${SRC_DATASETS[@]}" \
            --src_data_dirs  "${SRC_COLD_DIRS[@]}" \
            --target_dataset "$DATASET" \
            --output_dir     "$CKPT_DIR" \
            --epochs         "$PRETRAIN_EPOCHS" \
            --lr             "$PRETRAIN_LR" \
            --batch_size     "$BATCH_SIZE" \
            --val_ratio      "$VAL_RATIO" \
            --sample_size    "$SAMPLE_SIZE" \
            --seed           "$SEED"

        ds_mark stage1_pretrain
        echo "  ✓ Pretrain → $CKPT_DIR/best_model.pth"
    fi

    # ---- Stage 2: Fine-tune on pooled D_eff data ----------------------------
    if ds_done stage2_finetune; then
        echo "▶ [Stage 2] Fine-tune ... SKIPPED (already done)"
    else
        echo "▶ [Stage 2] Fine-tune on FinetuneData from: ${SRC_DATASETS[*]} ..."

        uv run python experiment/pretrain_multi.py \
            --phase           finetune \
            --src_datasets    "${SRC_DATASETS[@]}" \
            --src_data_dirs   "${SRC_FINETUNE_DIRS[@]}" \
            --target_dataset  "$DATASET" \
            --init_checkpoint "$CKPT_DIR/best_model.pth" \
            --output_dir      "$CKPT_DIR" \
            --epochs          "$FINETUNE_EPOCHS" \
            --lr              "$FINETUNE_LR" \
            --batch_size      "$BATCH_SIZE" \
            --val_ratio       "$VAL_RATIO" \
            --sample_size     "$SAMPLE_SIZE" \
            --seed            "$SEED"

        ds_mark stage2_finetune
        echo "  ✓ Fine-tune → $CKPT_DIR/ef_best_model.pth"
    fi

    # ---- Stage 3a: Generate test-set graphs ----------------------------------
    if ds_done stage3a_gen_graphs; then
        echo "▶ [Stage 3a] Generate test graphs ... SKIPPED (already done)"
    else
        echo "▶ [Stage 3a] Generate test-set graphs ($DATASET) ..."
        mkdir -p "$GRAPHS_DIR"

        uv run python experiment/generate_graphs.py \
            --model_path   "$CKPT_DIR" \
            --dataset      "$DATASET" \
            --dataset_path "$DATASET_JSON" \
            --output_file  "$GRAPHS_FILE" \
            --model_type   arg_designer \
            ${TASK_SPLIT_RAW:+--task_split_path "$PROJECT_ROOT/$TASK_SPLIT_RAW"} \
            ${LIMIT:+--limit "$LIMIT"} \
            $NO_EF_FLAG

        ds_mark stage3a_gen_graphs
        echo "  ✓ Test graphs → $GRAPHS_FILE"
    fi

    # ---- Stage 4a: Generate train-set graphs (overfitting check) -------------
    if ds_done stage4a_gen_graphs_train; then
        echo "▶ [Stage 4a] Generate train-set graphs ... SKIPPED (already done)"
    else
        echo "▶ [Stage 4a] Generate train-set graphs ($DATASET) ..."
        mkdir -p "$TRAIN_GRAPHS_DIR"

        uv run python experiment/generate_graphs.py \
            --model_path   "$CKPT_DIR" \
            --dataset      "$DATASET" \
            --dataset_path "$DATASET_JSON" \
            --output_file  "$TRAIN_GRAPHS_FILE" \
            --model_type   arg_designer_train \
            --split_key    "base_tasks_indices,finetune_tasks_indices" \
            ${TASK_SPLIT_RAW:+--task_split_path "$PROJECT_ROOT/$TASK_SPLIT_RAW"} \
            $NO_EF_FLAG

        ds_mark stage4a_gen_graphs_train
        echo "  ✓ Train graphs → $TRAIN_GRAPHS_FILE"
    fi
done

# ============================================================================
# Pass 2 (vLLM on): benchmark all datasets
# ============================================================================
echo ""
echo "════════════ Pass 2: Benchmark (all datasets) ════════════"
_ensure_vllm_running

for i in "${!DATASETS[@]}"; do
    DATASET="${DATASETS[$i]}"
    DECISION="${DECISION_METHODS[$i]}"

    GRAPHS_FILE="$EXP_BASE/graphs/arg_designer/${DATASET}_graphs.jsonl"
    RESULTS_DIR="$EXP_BASE/benchmark_results/pregraph/arg_designer"
    TRAIN_GRAPHS_FILE="$EXP_BASE/graphs/arg_designer_train/${DATASET}_graphs.jsonl"
    TRAIN_RESULTS_DIR="$EXP_BASE/benchmark_results/pregraph/arg_designer_train"

    _compute_source_datasets
    echo ""
    echo "  ── $DATASET  [${LOO_MODE^^} excluded: ${EXCLUDED_DISPLAY[*]}]"

    # ---- Stage 3b: Benchmark test set ----------------------------------------
    if ds_done stage3b_benchmark; then
        echo "▶ [Stage 3b] Benchmark test set ... SKIPPED (already done)"
    elif [[ ! -f "$GRAPHS_FILE" ]]; then
        echo "  WARNING: graphs file missing for $DATASET — skipping"
    else
        echo "▶ [Stage 3b] Benchmark test set ($DATASET) ..."
        mkdir -p "$RESULTS_DIR"

        uv run python experiment/benchmark_pregraph.py \
            --graphs_file      "$GRAPHS_FILE" \
            --dataset          "$DATASET" \
            --llm_name         "$HF_MODEL" \
            --decision_method  "$DECISION" \
            --output_file      "$RESULTS_DIR/${DATASET}.jsonl" \
            --summary_log_file "$RESULTS_DIR/summary.jsonl" \
            --eval_batch_size  "$EVAL_BATCH" \
            --model_type       arg_designer \
            ${LIMIT:+--limit "$LIMIT"}

        ds_mark stage3b_benchmark
        echo "  ✓ Results → $RESULTS_DIR/${DATASET}.jsonl"
    fi

    # ---- Stage 4b: Benchmark train set ---------------------------------------
    if ds_done stage4b_benchmark_train; then
        echo "▶ [Stage 4b] Benchmark train set ... SKIPPED (already done)"
    elif [[ ! -f "$TRAIN_GRAPHS_FILE" ]]; then
        echo "  WARNING: train graphs file missing for $DATASET — skipping"
    else
        echo "▶ [Stage 4b] Benchmark train set ($DATASET) ..."
        mkdir -p "$TRAIN_RESULTS_DIR"

        uv run python experiment/benchmark_pregraph.py \
            --graphs_file      "$TRAIN_GRAPHS_FILE" \
            --dataset          "$DATASET" \
            --llm_name         "$HF_MODEL" \
            --decision_method  "$DECISION" \
            --output_file      "$TRAIN_RESULTS_DIR/${DATASET}.jsonl" \
            --summary_log_file "$TRAIN_RESULTS_DIR/summary.jsonl" \
            --eval_batch_size  "$EVAL_BATCH" \
            --model_type       arg_designer_train

        ds_mark stage4b_benchmark_train
        echo "  ✓ Train results → $TRAIN_RESULTS_DIR/${DATASET}.jsonl"
    fi
done

echo ""
echo "════════════════════════════════════════════════"
echo "  ARGDesigner LOO/LGO pipeline complete"
echo "  Mode          : $LOO_MODE"
echo "  Experiment    : ${LLM_SLUG}/${EXP_NAME}/${RUN}/"
echo "  Test results  : ${LLM_SLUG}/${EXP_NAME}/${RUN}/benchmark_results/pregraph/arg_designer/summary.jsonl"
echo "  Train results : ${LLM_SLUG}/${EXP_NAME}/${RUN}/benchmark_results/pregraph/arg_designer_train/summary.jsonl"
echo "  Finished: $(date)"
echo "════════════════════════════════════════════════"
