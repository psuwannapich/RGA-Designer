#!/bin/bash
#SBATCH --job-name=install_vllm
#SBATCH --output=logs/install_vllm_%j.out
#SBATCH --error=logs/install_vllm_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --time=0-01:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=poomphob.suwannapichat@uni.lu
# ---------------------------------------------------------------------------
# One-time job to install vLLM into the project venv on a GPU node where
# CUDA_HOME is available.
#
# Usage:
#   sbatch slurm/install_vllm.sh
# ---------------------------------------------------------------------------

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p "$PROJECT_ROOT/logs"

cd "$PROJECT_ROOT"

echo "========================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "Started  : $(date)"
echo "========================================"

# ---- Locate CUDA -------------------------------------------------------
# Try common locations if CUDA_HOME is not already set.
if [[ -z "${CUDA_HOME:-}" ]]; then
    for candidate in /usr/local/cuda /usr/cuda $(dirname "$(dirname "$(which nvcc 2>/dev/null || true)")"); do
        if [[ -d "$candidate/lib64" ]]; then
            export CUDA_HOME="$candidate"
            break
        fi
    done
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
    echo "ERROR: Could not find CUDA_HOME. Try loading a CUDA module first:"
    echo "  module load CUDA && sbatch slurm/install_vllm.sh"
    exit 1
fi

echo "CUDA_HOME : $CUDA_HOME"
echo "nvcc      : $(nvcc --version | head -1)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# ---- Install vLLM -------------------------------------------------------
echo ""
echo "Installing vllm ..."
uv pip install "vllm>=0.6.1.post2"

echo ""
echo "Verifying install ..."
uv run python -c "import vllm; print(f'vllm {vllm.__version__} installed OK')"

echo "========================================"
echo "vLLM install complete."
echo "Finished : $(date)"
echo "========================================"
