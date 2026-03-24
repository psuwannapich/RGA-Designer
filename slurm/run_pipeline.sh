#!/bin/bash
# ---------------------------------------------------------------------------
# Full ARG-Designer pipeline: cold-start → train → benchmark
#
# Submits three dependent job arrays.  Each stage waits for all jobs in the
# previous stage to succeed before starting.
#
# Usage:
#   bash slurm/run_pipeline.sh
#
# Optional env vars (passed through to each stage):
#   HF_MODEL          HuggingFace model ID or Ollama name  (default: Qwen/Qwen3-8B)
#   NUM_TASKS         cold-start tasks per dataset          (default: 0)
#   NUM_ITERATIONS    iterations for train/test split size  (default: 10)
#   EPOCHS            ARGDesigner training epochs           (default: 100)
#   EVAL_BATCH        benchmark inference batch size        (default: 8)
#   DATASETS_ARRAY    Slurm array spec, e.g. "0-2" or "0,3" (default: 0-5 = all)
#   NUM_GPUS          GPUs per job for LLM stages           (default: 2, train always uses 1)
#                     Use 2+ V100-16GB instead of 1 V100-32GB
#   FINETUNE_EPOCHS   Phase-2 fine-tuning epochs            (default: 200)
#   FINETUNE_LR       Phase-2 learning rate                 (default: 5e-5)
#
# Example — run only gsm8k (index 0):
#   DATASETS_ARRAY=0 bash slurm/run_pipeline.sh
#
# Example — custom model, more tasks:
#   HF_MODEL=meta-llama/Llama-3.2-3B-Instruct NUM_TASKS=0 EPOCHS=200 \
#       bash slurm/run_pipeline.sh
# ---------------------------------------------------------------------------

set -euo pipefail

HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
NUM_TASKS="${NUM_TASKS:-0}"
EPOCHS="${EPOCHS:-100}"
EVAL_BATCH="${EVAL_BATCH:-8}"
DATASETS_ARRAY="${DATASETS_ARRAY:-0-5}"
NUM_GPUS="${NUM_GPUS:-2}"          # GPUs for LLM stages; train always uses 1
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-200}"
FINETUNE_LR="${FINETUNE_LR:-5e-5}"

COLD_START_ROOT="${COLD_START_ROOT:-ColdStartData}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
MODEL_SLUG="${HF_MODEL//\//-}"                     # Qwen/Qwen3-8B → Qwen-Qwen3-8B

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs"

echo "================================================"
echo "ARG-Designer Full Pipeline"
echo "Model         : $HF_MODEL"
echo "Datasets array: $DATASETS_ARRAY"
echo "Cold-start tasks per dataset: ${NUM_TASKS} (0 = all base tasks)"
echo "Training epochs: $EPOCHS"
echo "GPUs per LLM job: $NUM_GPUS  (train stage always uses 1)"
echo "================================================"

# ---- Stage 1: Cold-start ---------------------------------------------------
echo ""
echo "[Stage 1] Submitting cold-start jobs (array: $DATASETS_ARRAY) ..."
COLD_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --gres="gpu:${NUM_GPUS}" \
    --export=ALL,HF_MODEL="$HF_MODEL",NUM_TASKS="$NUM_TASKS" \
    "$PROJECT_ROOT/slurm/cold_start_all.sh" \
    | awk '{print $NF}')
echo "  Cold-start job ID: $COLD_JOB"

# ---- Stage 2: Train --------------------------------------------------------
echo ""
echo "[Stage 2] Submitting training jobs (depends on cold-start $COLD_JOB) ..."
TRAIN_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --dependency=afterok:"$COLD_JOB" \
    --gres="gpu:1" \
    --export=ALL,HF_MODEL="$HF_MODEL",COLD_START_ROOT="$COLD_START_ROOT",CHECKPOINT_ROOT="$CHECKPOINT_ROOT",EPOCHS="$EPOCHS" \
    "$PROJECT_ROOT/slurm/train.sh" \
    | awk '{print $NF}')
echo "  Train job ID: $TRAIN_JOB"

# ---- Stage 2.5: Fine-tune (Phase 2 — build D_eff + fine-tune) -------------
echo ""
echo "[Stage 2.5] Submitting fine-tune jobs (depends on training $TRAIN_JOB) ..."
FINETUNE_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --dependency=afterok:"$TRAIN_JOB" \
    --gres="gpu:${NUM_GPUS}" \
    --export=ALL,HF_MODEL="$HF_MODEL",COLD_START_ROOT="$COLD_START_ROOT",CHECKPOINT_ROOT="$CHECKPOINT_ROOT",FINETUNE_EPOCHS="$FINETUNE_EPOCHS",FINETUNE_LR="$FINETUNE_LR" \
    "$PROJECT_ROOT/slurm/finetune.sh" \
    | awk '{print $NF}')
echo "  Fine-tune job ID: $FINETUNE_JOB"

# ---- Stage 3: Benchmark ----------------------------------------------------
# benchmark.sh needs MODEL_PATH per dataset; use a wrapper that derives it
# from the array index at runtime.
echo ""
echo "[Stage 3] Submitting benchmark jobs (depends on fine-tune $FINETUNE_JOB) ..."

# Create a temporary wrapper that sets MODEL_PATH from the checkpoint root
BENCH_WRAPPER="$PROJECT_ROOT/logs/benchmark_wrapper_$$.sh"
cat > "$BENCH_WRAPPER" << WRAPPER_EOF
#!/bin/bash
#SBATCH --job-name=arg_bench_pipeline
#SBATCH --output=$PROJECT_ROOT/logs/bench_pipeline_%A_%a.out
#SBATCH --error=$PROJECT_ROOT/logs/bench_pipeline_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH -p gpu
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

set -euo pipefail

DATASETS=(gsm8k aqua multiarith svamp humaneval mmlu)
DATASET="\${DATASETS[\$SLURM_ARRAY_TASK_ID]}"
export MODEL_PATH="${MODEL_SLUG}/${CHECKPOINT_ROOT}/\${DATASET}"
export HF_MODEL="${HF_MODEL}"
export EVAL_BATCH="${EVAL_BATCH}"

bash "$PROJECT_ROOT/slurm/benchmark.sh"
WRAPPER_EOF

BENCH_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --dependency=afterok:"$FINETUNE_JOB" \
    --gres="gpu:${NUM_GPUS}" \
    "$BENCH_WRAPPER" \
    | awk '{print $NF}')
echo "  Benchmark job ID: $BENCH_JOB"

echo ""
echo "================================================"
echo "All stages submitted successfully."
echo ""
echo "  Stage 1   cold-start : job $COLD_JOB"
echo "  Stage 2   train      : job $TRAIN_JOB     (waits for $COLD_JOB)"
echo "  Stage 2.5 fine-tune  : job $FINETUNE_JOB  (waits for $TRAIN_JOB)"
echo "  Stage 3   benchmark  : job $BENCH_JOB     (waits for $FINETUNE_JOB)"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f $PROJECT_ROOT/logs/cold_start_${COLD_JOB}_*.out"
echo "  tail -f $PROJECT_ROOT/logs/train_${TRAIN_JOB}_*.out"
echo "  tail -f $PROJECT_ROOT/logs/finetune_${FINETUNE_JOB}_*.out"
echo "  tail -f $PROJECT_ROOT/logs/bench_pipeline_${BENCH_JOB}_*.out"
echo ""
echo "Results summary (after benchmark completes):"
echo "  cat $PROJECT_ROOT/logs/evaluation_summary.jsonl"
echo "================================================"
