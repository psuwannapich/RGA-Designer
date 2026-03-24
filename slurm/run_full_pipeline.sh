#!/bin/bash
# ---------------------------------------------------------------------------
# Full combined pipeline: ARG-Designer training → benchmark + RLHF (parallel)
#
# Execution order
# ───────────────
#   Stage 1    cold-start         ARG-Designer training data generation
#   Stage 2    train              ARG-Designer model pre-training
#   Stage 2.5  finetune           ARG-Designer Phase-2 fine-tuning
#
#   After finetune completes, the following two tracks run IN PARALLEL:
#
#   Track A (ARG-Designer benchmark)
#     Stage 3a  benchmark         ARG-Designer evaluation on all datasets
#     Stage 3b  benchmark_baselines  Baseline comparisons (all methods × datasets)
#
#   Track B (RLHF)
#     Stage 4   rlhf_collect      Collect preference pairs using fine-tuned model
#     Stage 5   rlhf_train_rm     Train reward model
#     Stage 6   rlhf_train_policy RLHF policy fine-tuning
#
# Usage
# ─────
#   bash slurm/run_full_pipeline.sh
#
# Optional env vars
# ─────────────────
#   HF_MODEL          HuggingFace model ID              (default: Qwen/Qwen3-8B)
#   DISABLE_THINKING  1 = skip <think> chain (Qwen3)    (default: 1)
#   DATASETS_ARRAY    Slurm array spec, e.g. "0-2"      (default: 0-5 = all)
#   NUM_GPUS          GPUs per LLM-inference job         (default: 2)
#                     train / train_rm always use 1 GPU
#
#   ARG-Designer knobs:
#   NUM_TASKS         cold-start tasks per dataset       (default: 0 = all)
#   EPOCHS            pre-training epochs                (default: 100)
#   FINETUNE_EPOCHS   Phase-2 fine-tuning epochs         (default: 200)
#   FINETUNE_LR       Phase-2 learning rate              (default: 5e-5)
#   EVAL_BATCH        benchmark inference batch size     (default: 8)
#   COLD_START_ROOT   cold-start data sub-dir            (default: ColdStartData)
#   CHECKPOINT_ROOT   checkpoint sub-dir                 (default: checkpoints)
#   RESULTS_ROOT      benchmark results sub-dir          (default: benchmark_results)
#
#   RLHF knobs:
#   RLHF_NUM_TASKS    tasks per dataset for collect      (default: 100)
#   PREFERENCE_ROOT   preference data sub-dir            (default: rlhf_data)
#   RM_ROOT           reward model checkpoint sub-dir    (default: rlhf_checkpoints)
#   POLICY_ROOT       policy checkpoint sub-dir          (default: rlhf_checkpoints)
#
# Example — run only gsm8k, use Llama, enable thinking:
#   HF_MODEL=meta-llama/Llama-3.2-3B-Instruct \
#   DISABLE_THINKING=0 DATASETS_ARRAY=0 \
#   bash slurm/run_full_pipeline.sh
# ---------------------------------------------------------------------------

set -euo pipefail

# ---- Configuration ----------------------------------------------------------
HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"
MODEL_SLUG="${HF_MODEL//\//-}"                  # Qwen/Qwen3-8B → Qwen-Qwen3-8B
DISABLE_THINKING="${DISABLE_THINKING:-1}"       # 1 = no-thinking mode (faster)
DATASETS_ARRAY="${DATASETS_ARRAY:-0-5}"
NUM_GPUS="${NUM_GPUS:-2}"

# ARG-Designer
NUM_TASKS="${NUM_TASKS:-0}"
EPOCHS="${EPOCHS:-100}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-200}"
FINETUNE_LR="${FINETUNE_LR:-5e-5}"
EVAL_BATCH="${EVAL_BATCH:-8}"
COLD_START_ROOT="${COLD_START_ROOT:-ColdStartData}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
RESULTS_ROOT="${RESULTS_ROOT:-benchmark_results}"

# RLHF
RLHF_NUM_TASKS="${RLHF_NUM_TASKS:-100}"
PREFERENCE_ROOT="${PREFERENCE_ROOT:-rlhf_data}"
RM_ROOT="${RM_ROOT:-rlhf_checkpoints}"
POLICY_ROOT="${POLICY_ROOT:-rlhf_checkpoints}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs"

echo "════════════════════════════════════════════════"
echo "  ARG-Designer + RLHF Combined Pipeline"
echo "════════════════════════════════════════════════"
echo "  Model            : $HF_MODEL  (slug: $MODEL_SLUG)"
echo "  Disable thinking : $DISABLE_THINKING"
echo "  Datasets array   : $DATASETS_ARRAY"
echo "  GPUs / LLM job   : $NUM_GPUS  (train stages use 1)"
echo ""
echo "  ARG-Designer"
echo "    cold-start tasks : ${NUM_TASKS} (0 = all base tasks)"
echo "    pre-train epochs : $EPOCHS"
echo "    finetune epochs  : $FINETUNE_EPOCHS  lr=$FINETUNE_LR"
echo ""
echo "  RLHF"
echo "    collect tasks    : $RLHF_NUM_TASKS"
echo "    preference root  : ${MODEL_SLUG}/${PREFERENCE_ROOT}"
echo "    rm/policy root   : ${MODEL_SLUG}/${RM_ROOT}"
echo "════════════════════════════════════════════════"
echo ""

# ---- Stage 1: Cold-start ----------------------------------------------------
echo "[Stage 1] Cold-start  (array: $DATASETS_ARRAY) ..."
COLD_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --gres="gpu:${NUM_GPUS}" \
    --export=ALL,HF_MODEL="$HF_MODEL",DISABLE_THINKING="$DISABLE_THINKING",NUM_TASKS="$NUM_TASKS" \
    "$PROJECT_ROOT/slurm/cold_start_all.sh" \
    | awk '{print $NF}')
echo "  job $COLD_JOB"

# ---- Stage 2: Pre-train -----------------------------------------------------
echo ""
echo "[Stage 2] Pre-train  (waits for $COLD_JOB) ..."
TRAIN_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --dependency=afterok:"$COLD_JOB" \
    --gres="gpu:1" \
    --export=ALL,HF_MODEL="$HF_MODEL",COLD_START_ROOT="$COLD_START_ROOT",CHECKPOINT_ROOT="$CHECKPOINT_ROOT",EPOCHS="$EPOCHS" \
    "$PROJECT_ROOT/slurm/train.sh" \
    | awk '{print $NF}')
echo "  job $TRAIN_JOB"

# ---- Stage 2.5: Fine-tune ---------------------------------------------------
echo ""
echo "[Stage 2.5] Fine-tune  (waits for $TRAIN_JOB) ..."
FINETUNE_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --dependency=afterok:"$TRAIN_JOB" \
    --gres="gpu:${NUM_GPUS}" \
    --export=ALL,HF_MODEL="$HF_MODEL",DISABLE_THINKING="$DISABLE_THINKING",COLD_START_ROOT="$COLD_START_ROOT",CHECKPOINT_ROOT="$CHECKPOINT_ROOT",FINETUNE_EPOCHS="$FINETUNE_EPOCHS",FINETUNE_LR="$FINETUNE_LR" \
    "$PROJECT_ROOT/slurm/finetune.sh" \
    | awk '{print $NF}')
echo "  job $FINETUNE_JOB"

# ---- Stage 3a: ARG-Designer benchmark (Track A) ----------------------------
# benchmark.sh derives MODEL_PATH from HF_MODEL + CHECKPOINT_ROOT + DATASET
# internally — no wrapper needed.
echo ""
echo "[Stage 3a] ARG-Designer benchmark  (waits for $FINETUNE_JOB) ..."
BENCH_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --dependency=afterok:"$FINETUNE_JOB" \
    --gres="gpu:${NUM_GPUS}" \
    --export=ALL,HF_MODEL="$HF_MODEL",DISABLE_THINKING="$DISABLE_THINKING",CHECKPOINT_ROOT="$CHECKPOINT_ROOT",RESULTS_ROOT="$RESULTS_ROOT",EVAL_BATCH="$EVAL_BATCH",VLLM_TENSOR_PARALLEL_SIZE="$NUM_GPUS" \
    "$PROJECT_ROOT/slurm/benchmark.sh" \
    | awk '{print $NF}')
echo "  job $BENCH_JOB"

# # ---- Stage 3b: Baseline benchmark (Track A, parallel with 3a) --------------
# echo ""
# echo "[Stage 3b] Baseline benchmark  (waits for $FINETUNE_JOB) ..."
# BASE_JOB=$(sbatch \
#     --array="$DATASETS_ARRAY" \
#     --dependency=afterok:"$FINETUNE_JOB" \
#     --gres="gpu:${NUM_GPUS}" \
#     --export=ALL,HF_MODEL="$HF_MODEL",DISABLE_THINKING="$DISABLE_THINKING",RESULTS_ROOT="$RESULTS_ROOT" \
#     "$PROJECT_ROOT/slurm/benchmark_baselines.sh" \
#     | awk '{print $NF}')
# echo "  job $BASE_JOB"

# ---- Stage 4: RLHF collect (Track B, also starts after finetune) -----------
echo ""
echo "[Stage 4] RLHF collect  (waits for $FINETUNE_JOB) ..."
COLLECT_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --dependency=afterok:"$FINETUNE_JOB" \
    --gres="gpu:${NUM_GPUS}" \
    --export=ALL,HF_MODEL="$HF_MODEL",DISABLE_THINKING="$DISABLE_THINKING",NUM_TASKS="$RLHF_NUM_TASKS",PREFERENCE_ROOT="$PREFERENCE_ROOT" \
    "$PROJECT_ROOT/slurm/collect.sh" \
    | awk '{print $NF}')
echo "  job $COLLECT_JOB"

# ---- Stage 5: RLHF train reward model --------------------------------------
echo ""
echo "[Stage 5] RLHF train_rm  (waits for $COLLECT_JOB) ..."
TRAIN_RM_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --dependency=afterok:"$COLLECT_JOB" \
    --gres="gpu:1" \
    --export=ALL,HF_MODEL="$HF_MODEL",PREFERENCE_ROOT="$PREFERENCE_ROOT",RM_ROOT="$RM_ROOT" \
    "$PROJECT_ROOT/slurm/train_rm.sh" \
    | awk '{print $NF}')
echo "  job $TRAIN_RM_JOB"

# ---- Stage 6: RLHF train policy --------------------------------------------
echo ""
echo "[Stage 6] RLHF train_policy  (waits for $TRAIN_RM_JOB) ..."
POLICY_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --dependency=afterok:"$TRAIN_RM_JOB" \
    --gres="gpu:${NUM_GPUS}" \
    --export=ALL,HF_MODEL="$HF_MODEL",DISABLE_THINKING="$DISABLE_THINKING",CHECKPOINT_ROOT="$CHECKPOINT_ROOT",RM_ROOT="$RM_ROOT",POLICY_ROOT="$POLICY_ROOT",NUM_TASKS="$RLHF_NUM_TASKS" \
    "$PROJECT_ROOT/slurm/train_policy.sh" \
    | awk '{print $NF}')
echo "  job $POLICY_JOB"

# ---- Stage 7: RLHF benchmark (after train_policy) --------------------------
echo ""
echo "[Stage 7] RLHF benchmark  (waits for $POLICY_JOB) ..."
RLHF_BENCH_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --dependency=afterok:"$POLICY_JOB" \
    --gres="gpu:${NUM_GPUS}" \
    --export=ALL,HF_MODEL="$HF_MODEL",DISABLE_THINKING="$DISABLE_THINKING",CHECKPOINT_ROOT="$POLICY_ROOT",RESULTS_ROOT="${RESULTS_ROOT}/rlhf",EVAL_BATCH="$EVAL_BATCH",VLLM_TENSOR_PARALLEL_SIZE="$NUM_GPUS" \
    "$PROJECT_ROOT/slurm/benchmark.sh" \
    | awk '{print $NF}')
echo "  job $RLHF_BENCH_JOB"

echo ""
echo "════════════════════════════════════════════════"
echo "  All stages submitted"
echo ""
echo "  [ARG-Designer training]"
echo "   1   cold-start  : $COLD_JOB"
echo "   2   train       : $TRAIN_JOB          ← waits for $COLD_JOB"
echo "   2.5 finetune    : $FINETUNE_JOB       ← waits for $TRAIN_JOB"
echo ""
echo "  [Track A — benchmarks, parallel after finetune]"
echo "   3a  benchmark   : $BENCH_JOB          ← waits for $FINETUNE_JOB"
echo "   3b  baselines   : $BASE_JOB           ← waits for $FINETUNE_JOB"
echo ""
echo "  [Track B — RLHF, parallel after finetune]"
echo "   4   collect     : $COLLECT_JOB        ← waits for $FINETUNE_JOB"
echo "   5   train_rm    : $TRAIN_RM_JOB       ← waits for $COLLECT_JOB"
echo "   6   train_policy: $POLICY_JOB         ← waits for $TRAIN_RM_JOB"
echo "   7   rlhf bench  : $RLHF_BENCH_JOB    ← waits for $POLICY_JOB"
echo ""
echo "  Monitor:  squeue -u \$USER"
echo "  Results (ARG-Designer): ${MODEL_SLUG}/${RESULTS_ROOT}/summary.jsonl"
echo "  Results (RLHF)        : ${MODEL_SLUG}/${RESULTS_ROOT}/rlhf/summary.jsonl"
echo "════════════════════════════════════════════════"
