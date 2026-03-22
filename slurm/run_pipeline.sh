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
#   NUM_TASKS         cold-start tasks per dataset          (default: 40)
#   EPOCHS            ARGDesigner training epochs           (default: 100)
#   EVAL_BATCH        benchmark inference batch size        (default: 8)
#   DATASETS_ARRAY    Slurm array spec, e.g. "0-2" or "0,3" (default: 0-5 = all)
#
# Example — run only gsm8k (index 0):
#   DATASETS_ARRAY=0 bash slurm/run_pipeline.sh
#
# Example — custom model, more tasks:
#   HF_MODEL=meta-llama/Llama-3.2-3B-Instruct NUM_TASKS=100 EPOCHS=200 \
#       bash slurm/run_pipeline.sh
# ---------------------------------------------------------------------------

set -euo pipefail

HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
NUM_TASKS="${NUM_TASKS:-40}"
EPOCHS="${EPOCHS:-100}"
EVAL_BATCH="${EVAL_BATCH:-8}"
DATASETS_ARRAY="${DATASETS_ARRAY:-0-5}"

COLD_START_ROOT="${COLD_START_ROOT:-ColdStartData_hf}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$PROJECT_ROOT/logs"

echo "================================================"
echo "ARG-Designer Full Pipeline"
echo "Model         : $HF_MODEL"
echo "Datasets array: $DATASETS_ARRAY"
echo "Cold-start tasks per dataset: $NUM_TASKS"
echo "Training epochs: $EPOCHS"
echo "================================================"

# ---- Stage 1: Cold-start ---------------------------------------------------
echo ""
echo "[Stage 1] Submitting cold-start jobs (array: $DATASETS_ARRAY) ..."
COLD_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
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
    --export=ALL,COLD_START_ROOT="$COLD_START_ROOT",CHECKPOINT_ROOT="$CHECKPOINT_ROOT",EPOCHS="$EPOCHS" \
    "$PROJECT_ROOT/slurm/train.sh" \
    | awk '{print $NF}')
echo "  Train job ID: $TRAIN_JOB"

# ---- Stage 3: Benchmark ----------------------------------------------------
# benchmark.sh needs MODEL_PATH per dataset; use a wrapper that derives it
# from the array index at runtime.
echo ""
echo "[Stage 3] Submitting benchmark jobs (depends on training $TRAIN_JOB) ..."

# Create a temporary wrapper that sets MODEL_PATH from the checkpoint root
BENCH_WRAPPER="$PROJECT_ROOT/logs/benchmark_wrapper_$$.sh"
cat > "$BENCH_WRAPPER" << WRAPPER_EOF
#!/bin/bash
#SBATCH --job-name=arg_bench_pipeline
#SBATCH --output=$PROJECT_ROOT/logs/bench_pipeline_%A_%a.out
#SBATCH --error=$PROJECT_ROOT/logs/bench_pipeline_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# #SBATCH --partition=gpu

set -euo pipefail

DATASETS=(gsm8k aqua multiarith svamp humaneval mmlu)
DATASET="\${DATASETS[\$SLURM_ARRAY_TASK_ID]}"
export MODEL_PATH="${CHECKPOINT_ROOT}/\${DATASET}"
export HF_MODEL="${HF_MODEL}"
export EVAL_BATCH="${EVAL_BATCH}"

bash "$PROJECT_ROOT/slurm/benchmark.sh"
WRAPPER_EOF

BENCH_JOB=$(sbatch \
    --array="$DATASETS_ARRAY" \
    --dependency=afterok:"$TRAIN_JOB" \
    "$BENCH_WRAPPER" \
    | awk '{print $NF}')
echo "  Benchmark job ID: $BENCH_JOB"

echo ""
echo "================================================"
echo "All stages submitted successfully."
echo ""
echo "  Stage 1 cold-start : job $COLD_JOB"
echo "  Stage 2 train      : job $TRAIN_JOB  (waits for $COLD_JOB)"
echo "  Stage 3 benchmark  : job $BENCH_JOB  (waits for $TRAIN_JOB)"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo "  tail -f $PROJECT_ROOT/logs/cold_start_${COLD_JOB}_*.out"
echo "  tail -f $PROJECT_ROOT/logs/train_${TRAIN_JOB}_*.out"
echo "  tail -f $PROJECT_ROOT/logs/bench_pipeline_${BENCH_JOB}_*.out"
echo ""
echo "Results summary (after benchmark completes):"
echo "  cat $PROJECT_ROOT/logs/evaluation_summary.jsonl"
echo "================================================"
