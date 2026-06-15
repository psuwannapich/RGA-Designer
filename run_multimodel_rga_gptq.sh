#!/bin/bash
#SBATCH --job-name=rga-mm-multiarith-gptq
#SBATCH --partition=gpu
#SBATCH --gres=gpu:volta:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rga_multimodel_gptq_%j.log
#SBATCH --error=logs/rga_multimodel_gptq_%j.log
# ---------------------------------------------------------------------------
# GPTQ-quantized variant of run_multimodel_rga.sh (multi-model RGA on
# multiarith). NOT submitted — prepared for future use.
#
# Differences vs run_multimodel_rga.sh:
#   - Each pool model is served from a pre-quantized GPTQ-Int4 checkpoint
#     (JunHowie/Qwen3-{4B,1.7B,0.6B}-GPTQ-Int4) instead of the fp16 weights.
#     --served-model-name still uses the original "Qwen/Qwen3-*" names so
#     MODEL_POOL / MODEL_ENDPOINTS / model_pool_qwen3.json and HF_MODEL all
#     stay unchanged — only the loaded weights differ.
#   - On V100 (sm_70, capability 70), vLLM's gptq_marlin kernel requires
#     capability >= 80 and is therefore never auto-selected; vLLM falls back
#     to the legacy "gptq" kernel (min_capability 60), which V100 supports.
#     --quantization gptq is passed explicitly to make this choice obvious.
#   - GPTQ-Int4 weights are ~3x smaller than fp16:
#       4B:   8.04 GB -> 2.67 GB
#       1.7B: 4.06 GB -> 1.36 GB
#       0.6B: 1.50 GB -> 0.54 GB
#       total 13.6 GB -> 4.57 GB
#     => all three models now fit comfortably on a SINGLE V100 (16GB) with
#     --max-model-len 8192 each, so this variant requests gpu:volta:1
#     instead of the fp16 script's gpu:volta:2.
#   - Output goes to a separate MODEL_SLUG so it never collides with the
#     fp16 multimodel run or the single-model production checkpoints.
#
# Caveat: the legacy gptq kernel dequantizes on the fly and is not
# Marlin-accelerated on V100, so per-token latency may be similar to (or
# slightly worse than) fp16 — the benefit here is GPU memory headroom
# (smaller footprint / fits on one GPU), not raw speed.
#
# To use: cancel/let finish the fp16 run, then `sbatch run_multimodel_rga_gptq.sh`.
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

# ---- Start vLLM servers: one process per pool model, all on GPU0 -----------
# GPTQ-Int4 weights total ~4.6GB, leaving ample headroom on a single V100 for
# --max-model-len 8192 KV cache on all three servers.
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

echo "[vLLM] Starting Qwen3-4B-GPTQ-Int4 on GPU0:$PORT_4B ..."
CUDA_VISIBLE_DEVICES=0 "$VLLM_SERVE_DIR/.venv/bin/vllm" serve JunHowie/Qwen3-4B-GPTQ-Int4 \
    --served-model-name Qwen/Qwen3-4B \
    --quantization gptq \
    --port "$PORT_4B" --dtype float16 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.35 --enforce-eager \
    --max-model-len 8192 \
    --chat-template "$VLLM_SERVE_DIR/qwen3_nonthinking.jinja" \
    > "logs/vllm_4b_gptq_${JOB_ID}.log" 2>&1 &
VLLM_PIDS+=($!)

echo "[vLLM] Starting Qwen3-1.7B-GPTQ-Int4 on GPU0:$PORT_1_7B ..."
CUDA_VISIBLE_DEVICES=0 "$VLLM_SERVE_DIR/.venv/bin/vllm" serve JunHowie/Qwen3-1.7B-GPTQ-Int4 \
    --served-model-name Qwen/Qwen3-1.7B \
    --quantization gptq \
    --port "$PORT_1_7B" --dtype float16 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.28 --enforce-eager \
    --max-model-len 8192 \
    --chat-template "$VLLM_SERVE_DIR/qwen3_nonthinking.jinja" \
    > "logs/vllm_1_7b_gptq_${JOB_ID}.log" 2>&1 &
VLLM_PIDS+=($!)

echo "[vLLM] Starting Qwen3-0.6B-GPTQ-Int4 on GPU0:$PORT_0_6B ..."
CUDA_VISIBLE_DEVICES=0 "$VLLM_SERVE_DIR/.venv/bin/vllm" serve JunHowie/Qwen3-0.6B-GPTQ-Int4 \
    --served-model-name Qwen/Qwen3-0.6B \
    --quantization gptq \
    --port "$PORT_0_6B" --dtype float16 --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.22 --enforce-eager \
    --max-model-len 8192 \
    --chat-template "$VLLM_SERVE_DIR/qwen3_nonthinking.jinja" \
    > "logs/vllm_0_6b_gptq_${JOB_ID}.log" 2>&1 &
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
export MODEL_SLUG="Qwen-Qwen3-4B-vllm-no_thinking-multimodel-gptq"  # isolated from fp16 multimodel + single-model checkpoints
export DATASET="multiarith"
export LOCAL_BASE_URL="http://localhost:${PORT_4B}/v1"
export LOCAL_API_KEY="EMPTY"

# Real distinct-weight pool: 3 vLLM servers (GPTQ-Int4), one per model.
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
echo "  scripts/train.sh  (multi-model GPTQ, $DATASET)"
echo "========================================"
bash scripts/train.sh

echo ""
echo "========================================"
echo "  scripts/rga.sh  (multi-model GPTQ, $DATASET)"
echo "========================================"
bash scripts/rga.sh

echo ""
echo "================================================"
echo "  Multi-model GPTQ RGA pipeline complete"
echo "  Slug: $MODEL_SLUG"
echo "  Finished: $(date)"
echo "================================================"
