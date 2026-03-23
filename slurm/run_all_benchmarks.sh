#!/bin/bash
# ---------------------------------------------------------------------------
# run_all_benchmarks.sh
#
# Submits two Slurm jobs in parallel:
#   1. benchmark.sh          — ARG-Designer (trained model) on all 6 datasets
#   2. benchmark_baselines.sh — all 8 baselines × 6 datasets
#
# Both jobs run concurrently; this script just prints the job IDs and exits.
#
# Optional env vars forwarded to both jobs:
#   CHECKPOINT_ROOT  finetune checkpoint root (default: checkpoints)
#                    benchmark.sh derives per-dataset path automatically
#   HF_MODEL         HuggingFace model ID     (default: Qwen/Qwen3-8B)
#   LIMIT            cap test samples         (default: all)
#   LLM_TIMEOUT      seconds per LLM call     (default: 600)
#   RESULTS_ROOT     output root directory    (default: benchmark_results)
#
# Optional env vars for baselines only:
#   NUM_AGENTS      agents per graph          (default: per-method default)
#   SC_SAMPLES      self-consistency samples  (default: 5)
#
# Optional env vars for ARGDesigner only:
#   EVAL_BATCH      evaluation batch size     (default: 8)
#
# Usage:
#   bash slurm/run_all_benchmarks.sh
#   CHECKPOINT_ROOT=my_checkpoints LIMIT=200 bash slurm/run_all_benchmarks.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Forward relevant env vars to sbatch -----------------------------------
SBATCH_VARS=(
    "HF_MODEL=${HF_MODEL:-Qwen/Qwen3-8B}"
)
[[ -n "${CHECKPOINT_ROOT:-}" ]] && SBATCH_VARS+=("CHECKPOINT_ROOT=${CHECKPOINT_ROOT}")
[[ -n "${LIMIT:-}"           ]] && SBATCH_VARS+=("LIMIT=${LIMIT}")
[[ -n "${LLM_TIMEOUT:-}"     ]] && SBATCH_VARS+=("LLM_TIMEOUT=${LLM_TIMEOUT}")
[[ -n "${RESULTS_ROOT:-}"    ]] && SBATCH_VARS+=("RESULTS_ROOT=${RESULTS_ROOT}")
[[ -n "${EVAL_BATCH:-}"      ]] && SBATCH_VARS+=("EVAL_BATCH=${EVAL_BATCH}")
[[ -n "${NUM_AGENTS:-}"      ]] && SBATCH_VARS+=("NUM_AGENTS=${NUM_AGENTS}")
[[ -n "${SC_SAMPLES:-}"      ]] && SBATCH_VARS+=("SC_SAMPLES=${SC_SAMPLES}")

# Build --export string
EXPORT_STR=$(IFS=","; echo "${SBATCH_VARS[*]}")

echo "Submitting benchmark jobs..."
echo "  CHECKPOINT_ROOT : ${CHECKPOINT_ROOT:-checkpoints}"
echo "  HF_MODEL        : ${HF_MODEL:-Qwen/Qwen3-8B}"
echo "  LIMIT           : ${LIMIT:-all}"
echo ""

# ---- Job 1: ARG-Designer ---------------------------------------------------
ARG_JID=$(sbatch \
    --export="${EXPORT_STR}" \
    --parsable \
    "${SCRIPT_DIR}/benchmark.sh")
echo "ARG-Designer job submitted  : ${ARG_JID}  (6 dataset tasks)"

# ---- Job 2: Baselines (7 methods × 6 datasets = 42 tasks) -----------------
BASE_JID=$(sbatch \
    --export="${EXPORT_STR}" \
    --parsable \
    "${SCRIPT_DIR}/benchmark_baselines.sh")
echo "Baseline methods job submitted: ${BASE_JID}  (48 method×dataset tasks)"

echo ""
echo "Both jobs running in parallel."
echo "Monitor with:  squeue -u \$USER"
echo "Results will be written to: \${RESULTS_ROOT:-benchmark_results}/"
echo "Summary log:                \${RESULTS_ROOT:-benchmark_results}/summary.jsonl"
