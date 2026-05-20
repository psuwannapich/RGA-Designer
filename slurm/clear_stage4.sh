#!/bin/bash
# ---------------------------------------------------------------------------
# clear_stage4.sh — Remove stage4a and stage4b artifacts from an experiment
# directory so the RLHF pipeline can re-run graph generation and benchmarking.
#
# Removes:
#   <exp_dir>/<run>/graphs/rlhf/                (stage4a output)
#   <exp_dir>/<run>/benchmark_results/pregraph/rlhf/  (stage4b output)
#   <exp_dir>/<run>/state/rlhf/*/stage4a_gen_graphs.done
#   <exp_dir>/<run>/state/rlhf/*/stage4b_benchmark.done
#
# Stages 1-3 sentinels and training artifacts are NOT touched.
#
# Usage:
#   bash slurm/clear_stage4.sh <exp_dir>
#   bash slurm/clear_stage4.sh Qwen-Qwen3-4B-exp_fix6_best_of_n_normal_temp-no_thinking
#
# The <exp_dir> can be an absolute path or a path relative to the project root
# (i.e. the directory containing this script's parent).
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <exp_dir>" >&2
    echo "  <exp_dir> can be absolute or relative to: $PROJECT_ROOT" >&2
    exit 1
fi

EXP_DIR="$1"

# Resolve relative paths against the project root
if [[ "$EXP_DIR" != /* ]]; then
    EXP_DIR="$PROJECT_ROOT/$EXP_DIR"
fi

if [[ ! -d "$EXP_DIR" ]]; then
    echo "ERROR: directory not found: $EXP_DIR" >&2
    exit 1
fi

echo "========================================"
echo "Clearing stage4a/4b artifacts from:"
echo "  $EXP_DIR"
echo "========================================"

# ---- stage4a: generated graphs ----------------------------------------------
COUNT=$(find "$EXP_DIR" -type d -name "rlhf" -path "*/graphs/rlhf" | wc -l)
find "$EXP_DIR" -type d -name "rlhf" -path "*/graphs/rlhf" -exec rm -rf {} + 2>/dev/null || true
echo "  graphs/rlhf dirs removed          : $COUNT"

# ---- stage4b: benchmark results ---------------------------------------------
COUNT=$(find "$EXP_DIR" -type d -name "rlhf" -path "*/benchmark_results/pregraph/rlhf" | wc -l)
find "$EXP_DIR" -type d -name "rlhf" -path "*/benchmark_results/pregraph/rlhf" -exec rm -rf {} + 2>/dev/null || true
echo "  benchmark_results/pregraph/rlhf dirs removed : $COUNT"

# ---- stage4 sentinel files --------------------------------------------------
COUNT=$(find "$EXP_DIR" \( -name "stage4a_gen_graphs.done" -o -name "stage4b_benchmark.done" \) | wc -l)
find "$EXP_DIR" \( -name "stage4a_gen_graphs.done" -o -name "stage4b_benchmark.done" \) -delete
echo "  stage4 sentinel files removed     : $COUNT"

# ---- Verification -----------------------------------------------------------
REMAINING=$(find "$EXP_DIR" -name "*.done" | wc -l)
echo "========================================"
echo "  Stage 1-3 sentinels still intact  : $REMAINING"
echo "  Done. Pipeline will resume from stage4a on next sbatch."
echo "========================================"
