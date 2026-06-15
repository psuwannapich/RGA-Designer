#!/bin/bash
#SBATCH --job-name=rga-mm-multiarith
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rga_multimodel_%j.log
#SBATCH --error=logs/rga_multimodel_%j.log
# ---------------------------------------------------------------------------
# Real (non-smoke) multi-model RGA run on multiarith.
#
# Starts 3 separate vLLM servers (one per MODEL_POOL member, real distinct
# weights) on 2 GPUs, then runs scripts/train.sh (cold-start -> pretrain ->
# finetune -> benchmark, with the model-selection head active) followed by
# scripts/rga.sh (collect -> train_rm -> train_policy -> benchmark) into a
# dedicated MODEL_SLUG so it never touches the existing single-model
# checkpoints under Qwen-Qwen3-4B-vllm-no_thinking/.
#
# Submit with:  sbatch run_multimodel_rga.sh
# ---------------------------------------------------------------------------
set -euo pipefail

VLLM_SERVE_DIR="/home/users/psuwannapichat/work_space/vllm_serve"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
mkdir -p logs

JOB_ID="${SLURM_JOB_ID:-$$}"
PORT_4B=8901
PORT_1_7B=8902
PORT_0_6B=8903

# ---- Start vLLM servers: one process per pool model, on its own GPU --------
# GPU0: Qwen3-4B alone (weights ~8GB, leaves headroom for --max-model-len 8192)
# GPU1: Qwen3-1.7B + Qwen3-0.6B together (combined weights ~4.6GB, both still
#       have ample room for --max-model-len 8192)
VLLM_PIDS=()

_cleanup() {
    echo ""
    echo "--- Cleanup: stopping vLLM servers ---"
    for pid in "${VLLM_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${VLLM_PIDS[@]}"; do
        wait "$pid" 2>/dev/null || true
    done
}
trap _cleanup EXIT

echo "[vLLM] Starting Qwen3-4B on GPU0:$PORT_4B ..."
CUDA_VISIBLE_DEVICES=0 "$VLLM_SERVE_DIR/.venv/bin/vllm" serve Qwen/Qwen3-4B \
    --served-model-name Qwen/Qwen3-4B \
    --port "$PORT_4B" --dtype float16 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 --enforce-eager \
    --max-model-len 8192 \
    --chat-template "$VLLM_SERVE_DIR/qwen3_nonthinking.jinja" \
    > "logs/vllm_4b_${JOB_ID}.log" 2>&1 &
VLLM_PIDS+=($!)

echo "[vLLM] Starting Qwen3-1.7B on GPU1:$PORT_1_7B ..."
CUDA_VISIBLE_DEVICES=1 "$VLLM_SERVE_DIR/.venv/bin/vllm" serve Qwen/Qwen3-1.7B \
    --served-model-name Qwen/Qwen3-1.7B \
    --port "$PORT_1_7B" --dtype float16 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.35 --enforce-eager \
    --max-model-len 8192 \
    --chat-template "$VLLM_SERVE_DIR/qwen3_nonthinking.jinja" \
    > "logs/vllm_1_7b_${JOB_ID}.log" 2>&1 &
VLLM_PIDS+=($!)

echo "[vLLM] Starting Qwen3-0.6B on GPU1:$PORT_0_6B ..."
CUDA_VISIBLE_DEVICES=1 "$VLLM_SERVE_DIR/.venv/bin/vllm" serve Qwen/Qwen3-0.6B \
    --served-model-name Qwen/Qwen3-0.6B \
    --port "$PORT_0_6B" --dtype float16 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.30 --enforce-eager \
    --max-model-len 8192 \
    --chat-template "$VLLM_SERVE_DIR/qwen3_nonthinking.jinja" \
    > "logs/vllm_0_6b_${JOB_ID}.log" 2>&1 &
VLLM_PIDS+=($!)

echo "  PIDs: ${VLLM_PIDS[*]}"
echo "  Waiting for all 3 servers to become healthy ..."

for PORT in "$PORT_4B" "$PORT_1_7B" "$PORT_0_6B"; do
    for i in $(seq 1 240); do
        if curl -sf "http://localhost:${PORT}/health" > /dev/null 2>&1; then
            echo "  Port $PORT ready after ${i}s."
            break
        fi
        if [[ $i -eq 240 ]]; then
            echo "  ERROR: server on port $PORT did not become ready in 240s."
            exit 1
        fi
        sleep 1
    done
done

# ---- Shared env: multi-model pool + per-model routing -----------------------
export HF_MODEL="Qwen/Qwen3-4B"
export MODEL_SLUG="Qwen-Qwen3-4B-vllm-no_thinking-multimodel"  # isolated from single-model checkpoints
export DATASET="multiarith"
export LOCAL_BASE_URL="http://localhost:${PORT_4B}/v1"
export LOCAL_API_KEY="EMPTY"

# Real distinct-weight pool: 3 vLLM servers, one per model.
export MODEL_POOL="$PROJECT_ROOT/model_pool_qwen3.json"
export MODEL_ENDPOINTS="{\"Qwen/Qwen3-4B\":\"http://localhost:${PORT_4B}/v1\",\"Qwen/Qwen3-1.7B\":\"http://localhost:${PORT_1_7B}/v1\",\"Qwen/Qwen3-0.6B\":\"http://localhost:${PORT_0_6B}/v1\"}"

# ---- scripts/train.sh overrides (medium scale) ------------------------------
export NUM_TASKS=0          # all base cold-start tasks (multiarith: 15)
export NUM_ITERATIONS=3     # vs prod default 10
export BATCH_SIZE=8
export EPOCHS=15            # vs prod default 30
export TRAIN_BATCH_SIZE=16
export FINETUNE_EPOCHS=15   # vs prod default 30
export EVAL_BATCH=8
export LIMIT=50             # benchmark on 50 test tasks (of 500)

# ---- scripts/rga.sh overrides (medium scale) ---------------------------------
export RLHF_NUM_TASKS=30        # vs prod default 100
export SAMPLE_TEMPERATURES="0.7 1.0 1.3"
export TASK_CONCURRENCY=6
export INFERENCE_CONCURRENCY=6
export RM_EPOCHS=20             # vs prod default 30
export POLICY_NUM_TASKS=30      # vs prod default 100
export POLICY_EPOCHS=15         # vs prod default 30
export BEST_OF_N=3              # vs prod default 5

# multiarith is ~98% solvable; keep "both correct" pairs (down-weighted)
# instead of excluding them entirely (rga.sh default BOTH_CORRECT_WEIGHT=0).
export BOTH_CORRECT_WEIGHT=0.1

# ---- Run pipeline ------------------------------------------------------------
echo ""
echo "========================================"
echo "  scripts/train.sh  (multi-model, $DATASET)"
echo "========================================"
bash scripts/train.sh

echo ""
echo "========================================"
echo "  scripts/rga.sh  (multi-model, $DATASET)"
echo "========================================"
bash scripts/rga.sh

echo ""
echo "================================================"
echo "  Multi-model RGA pipeline complete"
echo "  Slug: $MODEL_SLUG"
echo "  Finished: $(date)"
echo "================================================"
