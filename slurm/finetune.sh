#!/bin/bash
#SBATCH --job-name=arg_finetune
#SBATCH --output=logs/finetune_%A_%a.out
#SBATCH --error=logs/finetune_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:2
#SBATCH --array=0-5          # 0=gsm8k 1=aqua 2=multiarith 3=svamp 4=humaneval 5=mmlu
#SBATCH -p gpu
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu

# ---------------------------------------------------------------------------
# Phase-2 fine-tuning: build D_eff then fine-tune the Phase-1 ARGDesigner.
#
# Array index → dataset:
#   0  gsm8k
#   1  aqua
#   2  multiarith
#   3  svamp
#   4  humaneval
#   5  mmlu
#
# Required env vars (set by run_pipeline.sh or manually):
#   HF_MODEL          LLM used for D_eff inference
#   COLD_START_ROOT   root prefix for cold-start dirs  (default: ColdStartData_hf)
#   CHECKPOINT_ROOT   root prefix for checkpoint dirs  (default: checkpoints)
#
# Optional overrides:
#   FINETUNE_EPOCHS   fine-tuning epochs               (default: 200)
#   FINETUNE_LR       fine-tuning learning rate         (default: 5e-5)
#   PRUNING_RATIO     edge pruning fraction for D_pruned (default: 0.25)
#   REPLAY_RATIO      D_replay fraction of D_exp        (default: 0.3)
#   BATCH_SIZE        async inference batch size        (default: 2)
#   TRAIN_BATCH_SIZE  training batch size               (default: 32)
#
# Usage:
#   sbatch slurm/finetune.sh
#   sbatch --array=0 slurm/finetune.sh          # gsm8k only
# ---------------------------------------------------------------------------

set -euo pipefail

# Load CUDA modules so libcudnn.so is on LD_LIBRARY_PATH before torch imports.
_SLURM_DIR="${SLURM_SUBMIT_DIR:+${SLURM_SUBMIT_DIR}/slurm}"
source "${_SLURM_DIR:-$(dirname "${BASH_SOURCE[0]}")}/setup_cuda.sh"

DATASETS=(
    gsm8k       # 0
    aqua        # 1
    multiarith  # 2
    svamp       # 3
    humaneval   # 4
    mmlu        # 5
)
DATASET_JSONS=(
    "benchmark_datasets/gsm8k/gsm8k.jsonl"            # 0
    "benchmark_datasets/AQuA/AQuA.jsonl"              # 1
    "benchmark_datasets/MultiArith/MultiArith.json"   # 2
    "benchmark_datasets/SVAMP/SVAMP.json"             # 3
    "benchmark_datasets/humaneval/humaneval-py.jsonl" # 4
    "benchmark_datasets/MMLU/data"                    # 5
)

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT_ROOT/logs"

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

HF_MODEL="${HF_MODEL:-Qwen/Qwen3-8B}"
MODEL_SLUG="${HF_MODEL//\//-}"                     # Qwen/Qwen3-8B → Qwen-Qwen3-8B
COLD_START_ROOT="${COLD_START_ROOT:-ColdStartData}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-checkpoints}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-200}"
FINETUNE_LR="${FINETUNE_LR:-5e-5}"
PRUNING_RATIO="${PRUNING_RATIO:-0.25}"
REPLAY_RATIO="${REPLAY_RATIO:-0.3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
SEED="${SEED:-42}"

DATASET_JSON="$PROJECT_ROOT/${DATASET_JSONS[$SLURM_ARRAY_TASK_ID]}"
COLD_START_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${COLD_START_ROOT}/${DATASET}"
CHECKPOINT_DIR="$PROJECT_ROOT/${MODEL_SLUG}/${CHECKPOINT_ROOT}/${DATASET}"

echo "========================================"
echo "Job ID         : $SLURM_JOB_ID  (array task $SLURM_ARRAY_TASK_ID)"
echo "Node           : $SLURM_NODELIST"
echo "Dataset        : $DATASET"
echo "LLM            : $HF_MODEL"
echo "Cold-start dir : $COLD_START_DIR"
echo "Checkpoint dir : $CHECKPOINT_DIR"
echo "Finetune epochs: $FINETUNE_EPOCHS  |  LR: $FINETUNE_LR"
echo "Pruning ratio  : $PRUNING_RATIO  |  Replay ratio: $REPLAY_RATIO"
echo "Started at     : $(date)"
echo "========================================"

cd "$PROJECT_ROOT"

uv run python experiment/finetune_gemma.py \
    --dataset        "$DATASET" \
    --dataset_json   "$DATASET_JSON" \
    --cold_start_dir "$COLD_START_DIR" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --output_dir     "$CHECKPOINT_DIR" \
    --llm_name       "$HF_MODEL" \
    --finetune_epochs "$FINETUNE_EPOCHS" \
    --finetune_lr    "$FINETUNE_LR" \
    --pruning_ratio  "$PRUNING_RATIO" \
    --replay_ratio   "$REPLAY_RATIO" \
    --batch_size     "$BATCH_SIZE" \
    --train_batch_size "$TRAIN_BATCH_SIZE" \
    --seed           "$SEED"

echo "========================================"
echo "Fine-tuning complete for $DATASET"
echo "ef_best_model.pth saved to: $CHECKPOINT_DIR"
echo "Finished at: $(date)"
echo "========================================"
