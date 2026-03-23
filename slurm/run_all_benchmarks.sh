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
# Required:
#   MODEL_PATH   path to the trained ARGDesigner checkpoint directory
#                (relative to project root, or absolute)
#
# Optional env vars forwarded to both jobs:
#   HF_MODEL        HuggingFace model ID      (default: Qwen/Qwen3-8B)
#   LIMIT           cap test samples per job  (default: all)
#   LLM_TIMEOUT     seconds per LLM call      (default: 600)
#   RESULTS_ROOT    output root directory     (default: benchmark_results)
#
# Optional env vars for baselines only:
#   NUM_AGENTS      agents per graph          (default: per-method default)
#   SC_SAMPLES      self-consistency samples  (default: 5)
#
# Optional env vars for ARGDesigner only:
#   EVAL_BATCH      evaluation batch size     (default: 8)
#
# Usage:
#   MODEL_PATH=checkpoints/gsm8k bash slurm/run_all_benchmarks.sh
#   MODEL_PATH=checkpoints/gsm8k LIMIT=200 HF_MODEL=Qwen/Qwen3-8B \
#       bash slurm/run_all_benchmarks.sh
# ---------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "ERROR: MODEL_PATH must be set to the ARGDesigner checkpoint directory."
    echo "  e.g.: MODEL_PATH=checkpoints/gsm8k bash slurm/run_all_benchmarks.sh"
    exit 1
fi

# ---- Forward relevant env vars to sbatch -----------------------------------
SBATCH_VARS=(
    "MODEL_PATH=${MODEL_PATH}"
    "HF_MODEL=${HF_MODEL:-Qwen/Qwen3-8B}"
)
[[ -n "${LIMIT:-}"        ]] && SBATCH_VARS+=("LIMIT=${LIMIT}")
[[ -n "${LLM_TIMEOUT:-}"  ]] && SBATCH_VARS+=("LLM_TIMEOUT=${LLM_TIMEOUT}")
[[ -n "${RESULTS_ROOT:-}" ]] && SBATCH_VARS+=("RESULTS_ROOT=${RESULTS_ROOT}")
[[ -n "${EVAL_BATCH:-}"   ]] && SBATCH_VARS+=("EVAL_BATCH=${EVAL_BATCH}")
[[ -n "${NUM_AGENTS:-}"   ]] && SBATCH_VARS+=("NUM_AGENTS=${NUM_AGENTS}")
[[ -n "${SC_SAMPLES:-}"   ]] && SBATCH_VARS+=("SC_SAMPLES=${SC_SAMPLES}")

# Build --export string
EXPORT_STR=$(IFS=","; echo "${SBATCH_VARS[*]}")

echo "Submitting benchmark jobs..."
echo "  MODEL_PATH : ${MODEL_PATH}"
echo "  HF_MODEL   : ${HF_MODEL:-Qwen/Qwen3-8B}"
echo "  LIMIT      : ${LIMIT:-all}"
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
