#!/bin/bash
#SBATCH --job-name=rlhf_exp
#SBATCH --output=logs/rlhf_exp_%A_%a.out
#SBATCH --error=logs/rlhf_exp_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5          # 0=gsm8k 1=aqua 2=multiarith 3=svamp 4=humaneval 5=mmlu
#SBATCH -p gpu
#SBATCH --time=8:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# RLHF experiment pipeline — shares ARG artifacts, writes to its own exp dir.
#
# Prerequisite: pipeline_arg.sh must have completed for this LLM + dataset.
# Reads  from: <project>/<llm_slug>/arg/<arg_run>/         (checkpoints, cold-start data)
# Writes to:   <project>/<llm_slug>/<exp_name>/<run>/      (preference data, RM, policy, results)
#
# Running multiple RLHF experiments for the same LLM:
#   EXP_NAME=baseline RUN=0 sbatch --array=0-5 slurm/pipeline_rlhf_exp.sh
#   EXP_NAME=baseline RUN=1 sbatch --array=0-5 slurm/pipeline_rlhf_exp.sh   # repeat run 1
#   EXP_NAME=kl05 KL_COEFF=0.5 RUN=0 sbatch --array=0-5 slurm/pipeline_rlhf_exp.sh
#
# Stages (sequential within each array task):
#   1a gen_candidates   sample graph topologies from ARG policy (no vLLM)
#   1b collect_llm      score candidates with the LLM (vLLM required)
#   2  train_rm         train GNN reward model on preference pairs (no vLLM)
#   3  train_policy     GRPO fine-tune ARGDesigner policy (no vLLM)
#   4a gen_graphs       sample graphs from RLHF policy (no vLLM)
#   4b benchmark        evaluate graphs with the LLM (vLLM required)
#
# Array index → dataset:
#   0 gsm8k  1 aqua  2 multiarith  3 svamp  4 humaneval  5 mmlu
#
# Usage:
#   sbatch slurm/pipeline_rlhf_exp.sh
#   RUN=1 sbatch --array=0 slurm/pipeline_rlhf_exp.sh   # gsm8k, run 1
#
# Optional env vars:
#   HF_MODEL              HuggingFace model ID              (default: Qwen/Qwen3-4B)
#   LLM_SLUG              directory name for the LLM        (default: HF_MODEL with / → -)
#                         must match the LLM_SLUG used in pipeline_arg.sh
#   EXP_NAME              experiment label used as subdir   (default: baseline)
#   RUN                   repetition index 0-4              (default: 0)
#   ARG_RUN               which ARG run to read from        (default: same as RUN)
#   PORT_OFFSET           add to vLLM port base for collisions (default: 0)
#   DISABLE_THINKING      1 = no-thinking mode (Qwen3)      (default: 1)
#   RLHF_NUM_TASKS        tasks to sample for collect       (default: 100)
#   POLICY_NUM_TASKS      tasks for policy training         (default: 100)
#   W_CORRECT             correctness weight in score       (default: 0.6)
#   W_SIZE                size-efficiency weight            (default: 0.2)
#   W_EDGE                edge-density weight               (default: 0.2)
#   PAIR_MARGIN           min score gap to form a pair      (default: 0.05)
#   BOTH_WRONG_WEIGHT     RM loss weight for both-wrong pairs  (default: 0)
#   BOTH_CORRECT_WEIGHT   RM loss weight for both-correct pairs (default: 0)
#   RM_LOSS               RM training loss: bradley_terry|bce  (default: bradley_terry)
#   RM_EPOCHS             reward model training epochs      (default: 30)
#   POLICY_EPOCHS         policy fine-tune epochs           (default: 30)
#   POLICY_LR             policy learning rate              (default: 5e-6)
#   KL_COEFF              KL penalty coefficient            (default: 0.1)
#   LAMBDA_EFF            efficiency bonus weight added to RM reward in GRPO
#                         bonus = lambda_eff * mean(1-nodes/max, 1-edges/max_edges)
#                         (default: 0 = disabled, recommended 0.1-0.3)
#   SAMPLES_PER_TASK      graphs sampled per task (GRPO)    (default: 4)
#   BEST_OF_N             BoN graph selection at inference  (default: 5)
#   EVAL_BATCH            benchmark batch size              (default: 8)
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
DATASET_MIN_AGENTS=(2 2 2 2 2 2)
DATASET_MAX_AGENTS=(4 4 4 4 5 6)
DECISION_METHODS=(FinalRefer FinalRefer FinalRefer FinalRefer FinalWriteCode FinalRefer)

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$SLURM_ARRAY_TASK_ID]}"
TASK_SPLIT_RAW="${TASK_SPLIT_PATHS[$SLURM_ARRAY_TASK_ID]}"
MIN_AGENTS="${MIN_AGENTS:-${DATASET_MIN_AGENTS[$SLURM_ARRAY_TASK_ID]}}"
MAX_AGENTS="${MAX_AGENTS:-${DATASET_MAX_AGENTS[$SLURM_ARRAY_TASK_ID]}}"
DECISION="${DECISION_METHODS[$SLURM_ARRAY_TASK_ID]}"

# ---- Model configuration ----------------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"

if [[ "$HF_MODEL" == *"Llama"* ]] || [[ "$HF_MODEL" == *"llama"* ]]; then
    IS_LLAMA=1
else
    IS_LLAMA=0
fi

DISABLE_THINKING="${DISABLE_THINKING:-$([ "$IS_LLAMA" = "1" ] && echo 0 || echo 1)}"

# LLM_SLUG must match what pipeline_arg.sh used so the ARG artifacts are found.
LLM_SLUG="${LLM_SLUG:-${HF_MODEL//\//-}}"

# EXP_NAME: experiment label used as the directory name (no spaces/slashes).
# Use a descriptive name so results are self-documenting, e.g. "baseline", "kl05", "more_tasks".
EXP_NAME="${EXP_NAME:-baseline}"

# RUN: repetition index (0-4 for 5 runs). Each run is isolated under <exp_name>/<run>/.
RUN="${RUN:-0}"

# ARG_RUN: which ARG run (from pipeline_arg.sh) to use as the base policy.
# Defaults to the same index as RUN so run-0 RLHF uses run-0 ARG, etc.
ARG_RUN="${ARG_RUN:-${RUN}}"

export DISABLE_THINKING LLM_SLUG EXP_NAME PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---- Hyperparameters --------------------------------------------------------
SEED="${SEED:-42}"

# Collect
RLHF_NUM_TASKS="${RLHF_NUM_TASKS:-100}"
W_CORRECT="${W_CORRECT:-0.6}"
W_SIZE="${W_SIZE:-0.3}"
W_EDGE="${W_EDGE:-0.1}"
PAIR_MARGIN="${PAIR_MARGIN:-0.05}"
PRUNING_RATIO="${PRUNING_RATIO:-0.25}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-2}"
TASK_CONCURRENCY="${TASK_CONCURRENCY:-8}"
INFERENCE_CONCURRENCY="${INFERENCE_CONCURRENCY:-8}"
LLM_TIMEOUT="${LLM_TIMEOUT:-1200}"
SAMPLE_TEMPERATURES="${SAMPLE_TEMPERATURES:-0.5 1.0 1.5 2.0}"
ARG_MODEL_SAMPLES="${ARG_MODEL_SAMPLES:-}"
DIFFICULTY_FILTER="${DIFFICULTY_FILTER:-0}"
MIN_FAIL_RATE="${MIN_FAIL_RATE:-0.05}"
WEAK_BASELINES="${WEAK_BASELINES:-1}"
ROLE_SWEEP="${ROLE_SWEEP:-1}"
ROLE_SWEEP_TOPOLOGY="${ROLE_SWEEP_TOPOLOGY:-Chain}"
ROLE_SWEEP_N_AGENTS="${ROLE_SWEEP_N_AGENTS:-2}"
ROLE_SWEEP_MAX_COMBOS="${ROLE_SWEEP_MAX_COMBOS:-}"

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
#
#   <project>/<llm_slug>/
#     arg/<arg_run>/                ← read-only (created by pipeline_arg.sh)
#       ColdStartData/<dataset>/
#       checkpoints/<dataset>/
#     <exp_name>/<run>/             ← written by this script
#       rlhf_data/<dataset>/
#       rlhf_checkpoints/<dataset>/
#       graphs/rlhf/<dataset>_graphs.jsonl
#       benchmark_results/pregraph/rlhf/<dataset>.jsonl
#       state/<dataset>/

ARG_BASE="$PROJECT_ROOT/${LLM_SLUG}/arg/${ARG_RUN}"
RLHF_BASE="$PROJECT_ROOT/${LLM_SLUG}/${EXP_NAME}/${RUN}"

# Paths read from the shared ARG run
ARG_CHECKPOINT_DIR="$ARG_BASE/checkpoints/${DATASET}"
ARG_COLD_START_DIR="$ARG_BASE/ColdStartData/${DATASET}"
ARG_D_EFF_DIR="$ARG_CHECKPOINT_DIR/FinetuneData_${DATASET}"

# Paths written by this RLHF run
PREFERENCE_DIR="$RLHF_BASE/rlhf_data/${DATASET}"
RM_CHECKPOINT="$RLHF_BASE/rlhf_checkpoints/${DATASET}/reward_model.pth"
POLICY_CHECKPOINT="$RLHF_BASE/rlhf_checkpoints/${DATASET}/policy_rlhf.pth"
GRAPHS_DIR="$RLHF_BASE/graphs/rlhf"
GRAPHS_FILE="$GRAPHS_DIR/${DATASET}_graphs.jsonl"
RESULTS_DIR="$RLHF_BASE/benchmark_results/pregraph/rlhf"
STATE_DIR="$RLHF_BASE/state/${DATASET}"
mkdir -p "$STATE_DIR"

# ---- vLLM inference server --------------------------------------------------
USE_VLLM_SERVER="${USE_VLLM_SERVER:-1}"
# Port range 9820-9825 + RUN*10 for RLHF experiments (avoids clashing with pipeline_arg.sh ports 7800-7849).
# Set PORT_OFFSET (e.g. 100, 200) when running different EXP_NAME jobs concurrently on the same node.
PORT_OFFSET="${PORT_OFFSET:-0}"
VLLM_PORT="${VLLM_PORT:-$((9820 + ${SLURM_ARRAY_TASK_ID:-0} + ${RUN} * 10 + PORT_OFFSET))}"
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
    mkdir -p "$PROJECT_ROOT/logs"
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
    echo "  ✓ vLLM server stopped (EXIT trap)."
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
            echo "  WARNING: vLLM still alive after 60s — sending SIGKILL ..."
            kill -9 "${VLLM_PID}" 2>/dev/null || true
            break
        fi
    done
    VLLM_PID=""
    echo "  ✓ vLLM server stopped. Waiting 5s for GPU VRAM to be reclaimed ..."
    sleep 5
}

# ---- Stage checkpointing ----------------------------------------------------
# Delete a .done file to re-run that stage:
#   rm <llm_slug>/rlhf/<run>/state/<dataset>/stage1a_gen_candidates.done
stage_done() { [[ -f "$STATE_DIR/$1.done" ]]; }
mark_done()  { touch "$STATE_DIR/$1.done"; echo "  ✔ checkpoint: $STATE_DIR/$1.done"; }

echo "════════════════════════════════════════════════"
echo "  RLHF Experiment Pipeline"
echo "  Job $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "  Node              : $SLURM_NODELIST"
echo "  Dataset           : $DATASET"
echo "  Model             : $HF_MODEL"
echo "  LLM slug          : $LLM_SLUG"
echo "  Experiment        : $EXP_NAME"
echo "  Run               : $RUN  (ARG run: $ARG_RUN)"
echo "  ARG base (read)   : ${LLM_SLUG}/arg/${ARG_RUN}/"
echo "  RLHF base (write) : ${LLM_SLUG}/${EXP_NAME}/${RUN}/"
echo "  Collect tasks     : $RLHF_NUM_TASKS"
echo "  Policy tasks      : $POLICY_NUM_TASKS"
echo "  Scoring weights   : correct=$W_CORRECT  size=$W_SIZE  edge=$W_EDGE  margin=$PAIR_MARGIN"
echo "  RM loss           : $RM_LOSS"
echo "  RM pair weights   : both_wrong=$BOTH_WRONG_WEIGHT  both_correct=$BOTH_CORRECT_WEIGHT"
echo "  RM epochs         : $RM_EPOCHS  lr=$RM_LR"
echo "  Policy epochs     : $POLICY_EPOCHS  lr=$POLICY_LR  kl=$KL_COEFF  lambda_eff=$LAMBDA_EFF"
echo "  Best-of-N         : $([ "${BEST_OF_N:-1}" -gt 1 ] && echo "N=$BEST_OF_N  temp=$BON_TEMPERATURE" || echo "disabled")"
echo "  vLLM server       : $([ "$USE_VLLM_SERVER" = "1" ] && echo "enabled (port $VLLM_PORT, tp=$VLLM_TP)" || echo "disabled (HF backend)")"
echo "  Started at        : $(date)"
echo "════════════════════════════════════════════════"

# Sanity check — ARG checkpoint must exist before starting.
if [[ ! -f "$ARG_CHECKPOINT_DIR/ef_best_model.pth" ]]; then
    echo "ERROR: ARG checkpoint not found: $ARG_CHECKPOINT_DIR/ef_best_model.pth"
    echo "  Run pipeline_arg.sh first (LLM_SLUG=$LLM_SLUG, RUN=$ARG_RUN, dataset=$DATASET)."
    exit 1
fi

if [[ "$USE_VLLM_SERVER" == "1" ]]; then
    export LOCAL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
    export LOCAL_API_KEY="EMPTY"
    export USE_VLLM_SERVER USE_VLLM=0
fi

# ---- Stage 1a: Generate ARGDesigner candidates (no vLLM) --------------------
echo ""
if stage_done stage1a_gen_candidates || stage_done stage1_collect; then
    echo "▶ [Stage 1a/4] Gen candidates ... SKIPPED (already done)"
else
    echo "▶ [Stage 1a/4] Gen candidates (ARGDesigner GNN, no vLLM) ..."
    _ensure_vllm_stopped

    # Build coldstart_dirs: point at the shared ARG cold-start and D_eff data.
    _coldstart_dirs=""
    [[ -d "$ARG_COLD_START_DIR" ]]          && _coldstart_dirs+=" $ARG_COLD_START_DIR"
    [[ -d "$ARG_D_EFF_DIR" ]]               && _coldstart_dirs+=" $ARG_D_EFF_DIR"
    [[ -d "$ARG_COLD_START_DIR/rlhf_rejected" ]] && _coldstart_dirs+=" $ARG_COLD_START_DIR/rlhf_rejected"
    [[ -d "$ARG_D_EFF_DIR/rlhf_rejected" ]]      && _coldstart_dirs+=" $ARG_D_EFF_DIR/rlhf_rejected"
    _coldstart_dirs="${_coldstart_dirs# }"  # trim leading space

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
        --arg_model_dir      "$ARG_CHECKPOINT_DIR" \
        ${ARG_MODEL_SAMPLES:+--arg_model_samples "$ARG_MODEL_SAMPLES"} \
        $([ "${DIFFICULTY_FILTER}" = "1" ] && echo "--difficulty_filter") \
        $([ "${DIFFICULTY_FILTER}" = "1" ] && echo "--min_fail_rate $MIN_FAIL_RATE") \
        $([ "${WEAK_BASELINES}"    = "1" ] && echo "--weak_baselines") \
        $([ "${ROLE_SWEEP}"        = "1" ] && echo "--role_sweep") \
        $([ "${ROLE_SWEEP}"        = "1" ] && echo "--role_sweep_topology $ROLE_SWEEP_TOPOLOGY") \
        $([ "${ROLE_SWEEP}"        = "1" ] && echo "--role_sweep_n_agents $ROLE_SWEEP_N_AGENTS") \
        ${ROLE_SWEEP_MAX_COMBOS:+$([ "${ROLE_SWEEP}" = "1" ] && echo "--role_sweep_max_combos $ROLE_SWEEP_MAX_COMBOS")} \
        ${_coldstart_dirs:+--coldstart_dirs $_coldstart_dirs}

    mark_done stage1a_gen_candidates
    echo "  ✓ Candidates → $PREFERENCE_DIR/candidates.pkl"
fi

# ---- Stage 1b: LLM scoring of candidates (vLLM required) --------------------
echo ""
if stage_done stage1b_collect_llm || stage_done stage1_collect; then
    echo "▶ [Stage 1b/4] Collect LLM scoring ... SKIPPED (already done)"
else
    echo "▶ [Stage 1b/4] Collect LLM scoring ..."
    _ensure_vllm_running

    # Same coldstart_dirs as 1a (for free pairs from pre-scored PT files).
    _coldstart_dirs=""
    [[ -d "$ARG_COLD_START_DIR" ]]          && _coldstart_dirs+=" $ARG_COLD_START_DIR"
    [[ -d "$ARG_D_EFF_DIR" ]]               && _coldstart_dirs+=" $ARG_D_EFF_DIR"
    [[ -d "$ARG_COLD_START_DIR/rlhf_rejected" ]] && _coldstart_dirs+=" $ARG_COLD_START_DIR/rlhf_rejected"
    [[ -d "$ARG_D_EFF_DIR/rlhf_rejected" ]]      && _coldstart_dirs+=" $ARG_D_EFF_DIR/rlhf_rejected"
    _coldstart_dirs="${_coldstart_dirs# }"

    uv run rlhf \
        --dataset               "$DATASET" \
        --phase                 collect_llm \
        --llm_name              "$HF_MODEL" \
        --dataset_json          "$DATASET_JSON" \
        --num_tasks             "$RLHF_NUM_TASKS" \
        --preference_dir        "$PREFERENCE_DIR" \
        --min_agents            "$MIN_AGENTS" \
        --max_agents            "$MAX_AGENTS" \
        --w_correct             "$W_CORRECT" \
        --w_size                "$W_SIZE" \
        --w_edge                "$W_EDGE" \
        --pair_margin           "$PAIR_MARGIN" \
        --pruning_ratio         "$PRUNING_RATIO" \
        --checkpoint_every      "$CHECKPOINT_EVERY" \
        --task_concurrency      "$TASK_CONCURRENCY" \
        --inference_concurrency "$INFERENCE_CONCURRENCY" \
        --llm_timeout           "$LLM_TIMEOUT" \
        --seed                  "$SEED" \
        ${_coldstart_dirs:+--coldstart_dirs $_coldstart_dirs}

    mark_done stage1b_collect_llm
    echo "  ✓ Preference pairs → $PREFERENCE_DIR"
fi

# ---- Stage 2: Train reward model (no vLLM) ----------------------------------
echo ""
if stage_done stage2_train_rm; then
    echo "▶ [Stage 2/4] Train reward model ... SKIPPED (already done)"
else
    echo "▶ [Stage 2/4] Train reward model ..."
    _ensure_vllm_stopped

    uv run rlhf \
        --dataset             "$DATASET" \
        --phase               train_rm \
        --preference_dir      "$PREFERENCE_DIR" \
        --rm_checkpoint       "$RM_CHECKPOINT" \
        --rm_epochs           "$RM_EPOCHS" \
        --rm_lr               "$RM_LR" \
        --rm_batch_size       "$RM_BATCH_SIZE" \
        --rm_hidden_dim       "$RM_HIDDEN_DIM" \
        --rm_output_dim       "$RM_OUTPUT_DIM" \
        --rm_val_fraction     "$RM_VAL_FRACTION" \
        --both_wrong_weight   "$BOTH_WRONG_WEIGHT" \
        --both_correct_weight "$BOTH_CORRECT_WEIGHT" \
        --rm_loss             "$RM_LOSS"

    mark_done stage2_train_rm
    echo "  ✓ Reward model → $RM_CHECKPOINT"
fi

# ---- Stage 3: Train RLHF policy (no vLLM) -----------------------------------
echo ""
if stage_done stage3_train_policy; then
    echo "▶ [Stage 3/4] Train RLHF policy ... SKIPPED (already done)"
else
    echo "▶ [Stage 3/4] Train RLHF policy ..."

    uv run rlhf \
        --dataset            "$DATASET" \
        --phase              train_policy \
        --llm_name           "$HF_MODEL" \
        --model_dir          "$ARG_CHECKPOINT_DIR" \
        --rm_checkpoint      "$RM_CHECKPOINT" \
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

    mark_done stage3_train_policy
    echo "  ✓ Policy → $POLICY_CHECKPOINT"
fi

# ---- Stage 4a: Generate graphs (RLHF policy, no vLLM) -----------------------
echo ""
NO_EF_FLAG=""
[[ "$NO_EF" == "1" ]] && NO_EF_FLAG="--no_ef"

# Best-of-N: use the reward model trained in this run for BoN ranking.
BON_FLAGS=""
if [[ "${BEST_OF_N:-1}" -gt 1 ]]; then
    ROLE_EMB_FILE="$ARG_COLD_START_DIR/precomputed_role_embeddings.pkl"
    if [[ ! -f "$RM_CHECKPOINT" ]]; then
        echo "WARNING: BEST_OF_N=$BEST_OF_N but RM checkpoint not found — disabling BoN"
    elif [[ ! -f "$ROLE_EMB_FILE" ]]; then
        echo "WARNING: BEST_OF_N=$BEST_OF_N but role embeddings not found: $ROLE_EMB_FILE — disabling BoN"
    else
        BON_FLAGS="--best_of_n $BEST_OF_N --bon_temperature $BON_TEMPERATURE --rm_checkpoint $RM_CHECKPOINT --role_emb_path $ROLE_EMB_FILE"
        echo "  [BoN] Enabled: N=$BEST_OF_N  temperature=$BON_TEMPERATURE"
    fi
fi

POLICY_DIR="$(dirname "$POLICY_CHECKPOINT")"

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
        --model_type   rlhf \
        ${TASK_SPLIT_RAW:+--task_split_path "$PROJECT_ROOT/$TASK_SPLIT_RAW"} \
        ${LIMIT:+--limit "$LIMIT"} \
        $NO_EF_FLAG \
        $BON_FLAGS

    mark_done stage4a_gen_graphs
    echo "  ✓ Graphs → $GRAPHS_FILE"
fi

# ---- Stage 4b: Benchmark (vLLM required) ------------------------------------
echo ""
if stage_done stage4b_benchmark; then
    echo "▶ [Stage 4b/4] Benchmark ... SKIPPED (already done)"
else
    echo "▶ [Stage 4b/4] Benchmark (RLHF graphs) ..."
    _ensure_vllm_running

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

    mark_done stage4b_benchmark
    echo "  ✓ Results → $OUTPUT_FILE"
fi

# ---- Stage 5a: Generate graphs on TRAINING set (overfitting check) ----------
echo ""
TRAIN_GRAPHS_DIR="$RLHF_BASE/graphs/rlhf_train"
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

    TRAIN_RESULTS_DIR="$RLHF_BASE/benchmark_results/pregraph/rlhf_train"
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
echo "  RLHF experiment complete for $DATASET"
echo "  ARG base  : ${LLM_SLUG}/arg/${ARG_RUN}/"
echo "  RLHF base : ${LLM_SLUG}/${EXP_NAME}/${RUN}/"
echo "  Test  results : ${LLM_SLUG}/${EXP_NAME}/${RUN}/benchmark_results/pregraph/rlhf/summary.jsonl"
echo "  Train results : ${LLM_SLUG}/${EXP_NAME}/${RUN}/benchmark_results/pregraph/rlhf_train/summary.jsonl"
echo "  Finished  : $(date)"
echo "════════════════════════════════════════════════"
