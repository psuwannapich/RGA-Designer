#!/bin/bash
# ---------------------------------------------------------------------------
# Smoke-test: start vLLM then drive scripts/train.sh and scripts/rga.sh
# with tiny overrides to verify the full refactored pipeline end-to-end.
#
# Usage:
#   bash smoke_test.sh
#   HF_MODEL=Qwen/Qwen3-4B VLLM_PORT=9988 bash smoke_test.sh
#
# Options (env vars):
#   HF_MODEL    HuggingFace model ID  (default: Qwen/Qwen3-4B)
#   VLLM_PORT   port for the vLLM server (default: 8899)
#   KEEP_DATA   1 = keep test artifacts under smoke-test/ after run
# ---------------------------------------------------------------------------
set -euo pipefail

VLLM_SERVE_DIR="/home/users/psuwannapichat/work_space/vllm_serve"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

HF_MODEL="${HF_MODEL:-Qwen/Qwen3-4B}"
VLLM_PORT="${VLLM_PORT:-8899}"
KEEP_DATA="${KEEP_DATA:-0}"
VLLM_LOG="/tmp/vllm_smoke_$$.log"
VLLM_PID=""

# ---- Cleanup ----------------------------------------------------------------
_cleanup() {
    echo ""
    echo "--- Cleanup ---"
    if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "  Stopping vLLM (pid $VLLM_PID) ..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        echo "  vLLM stopped."
    fi
    if [[ "$KEEP_DATA" != "1" && -d "$PROJECT_ROOT/smoke-test" ]]; then
        echo "  Removing test artifacts: smoke-test/"
        rm -rf "$PROJECT_ROOT/smoke-test"
    fi
    rm -f "$VLLM_LOG"
}
trap _cleanup EXIT

# ---- Banner -----------------------------------------------------------------
echo "================================================"
echo "  ARG/RGA Designer — Smoke Test"
echo "  Model   : $HF_MODEL"
echo "  Port    : $VLLM_PORT"
echo "  Started : $(date)"
echo "================================================"

# ---- Start vLLM -------------------------------------------------------------
echo ""
echo "[vLLM] Starting server ..."

# --served-model-name aliases let one loaded model answer requests for every
# name in MODEL_POOL, so the multi-model code path (per-node model selection,
# routing, embeddings) is exercised without needing N separate model weights.
"$VLLM_SERVE_DIR/.venv/bin/vllm" serve "$HF_MODEL" \
    --served-model-name "$HF_MODEL" Qwen/Qwen3-1.7B Qwen/Qwen3-0.6B \
    --port         "$VLLM_PORT" \
    --dtype        float16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.85 \
    --enforce-eager \
    --max-model-len 4096 \
    --chat-template "$VLLM_SERVE_DIR/qwen3_nonthinking.jinja" \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

echo "  PID: $VLLM_PID  (log: $VLLM_LOG)"
echo "  Waiting for health ..."

for i in $(seq 1 180); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "  ERROR: vLLM exited early. Last log:"
        tail -20 "$VLLM_LOG" >&2
        exit 1
    fi
    if curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo "  Ready after ${i}s."
        break
    fi
    if [[ $i -eq 180 ]]; then
        echo "  ERROR: vLLM did not become ready in 180s. Last log:"
        tail -20 "$VLLM_LOG" >&2
        exit 1
    fi
    sleep 1
done

# ---- Shared env overrides ---------------------------------------------------
# Passed to both scripts via export; each script picks up what it needs.
export HF_MODEL
export MODEL_SLUG="smoke-test"         # isolated slug — no real data touched
export DATASET="multiarith"
export LOCAL_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export LOCAL_API_KEY="EMPTY"

# Multi-model pool — must match the --served-model-name aliases above so
# every pool member resolves against the single loaded vLLM model.
export MODEL_POOL="${MODEL_POOL:-$HF_MODEL,Qwen/Qwen3-1.7B,Qwen/Qwen3-0.6B}"

# train.sh overrides
export NUM_TASKS=5                     # cold-start tasks
export NUM_ITERATIONS=1                # iterations per topology
export BATCH_SIZE=2                    # LLM batch
export NUM_ROUNDS=1
export MIN_AGENTS=2
export MAX_AGENTS=3
export EPOCHS=2                        # pretrain epochs
export TRAIN_BATCH_SIZE=4
export FINETUNE_EPOCHS=2              # curriculum fine-tune epochs
export LIMIT=3                         # benchmark: evaluate 3 test tasks only
export EVAL_BATCH=2

# rga.sh / rga_global_rm.sh overrides
export RLHF_NUM_TASKS=5               # preference collection tasks
export SAMPLE_TEMPERATURES="0.5 1.0"  # fewer temperature samples
export TASK_CONCURRENCY=2
export INFERENCE_CONCURRENCY=2
export WEAK_BASELINES=0               # skip weak-baseline graphs
export ROLE_SWEEP=0                   # skip role sweep
export RM_EPOCHS=5
export RM_BATCH_SIZE=4
export POLICY_NUM_TASKS=5
export POLICY_EPOCHS=2
export SAMPLES_PER_TASK=2
export GRAD_ACCUM_STEPS=2
export BEST_OF_N=1                    # disable BoN for speed

# MultiArith is easy (~98% baseline accuracy), so with only RLHF_NUM_TASKS=5
# most sampled graphs are "both correct" — rga.sh's default
# BOTH_CORRECT_WEIGHT=0 excludes those pairs entirely, which can leave the
# reward-model dataset empty. Keep them (down-weighted) instead, matching
# rga_global_rm.sh's default.
export BOTH_CORRECT_WEIGHT=0.1

# rga_global_rm.sh overrides
export RUN_NAME="smoke"               # namespaces global-RM outputs

# ---- Run scripts/train.sh ---------------------------------------------------
echo ""
echo "========================================"
echo "  scripts/train.sh  (Stages 1-4)"
echo "========================================"
bash scripts/train.sh
echo ""
echo "  [PASS] scripts/train.sh"

# ---- Run scripts/rga.sh -----------------------------------------------------
echo ""
echo "========================================"
echo "  scripts/rga.sh  (RGA Stages 1-4)"
echo "========================================"
bash scripts/rga.sh
echo ""
echo "  [PASS] scripts/rga.sh"

# ---- Run scripts/rga_global_rm.sh -------------------------------------------
echo ""
echo "========================================"
echo "  scripts/rga_global_rm.sh  (Global RM)"
echo "========================================"
bash scripts/rga_global_rm.sh
echo ""
echo "  [PASS] scripts/rga_global_rm.sh"

# ---- Done -------------------------------------------------------------------
echo ""
echo "================================================"
echo "  ALL STAGES PASSED"
echo "  Finished: $(date)"
echo "================================================"
