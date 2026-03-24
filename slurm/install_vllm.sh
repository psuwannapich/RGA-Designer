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
# One-time job to install vLLM into the project venv on a GPU node.
# Usage:  sbatch slurm/install_vllm.sh
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

# ---- Try to enable the module system (Lmod / Environment Modules) --------
# Temporarily disable -u so Lmod init scripts can reference unset variables.
set +u
for _init in \
    /usr/share/lmod/lmod/init/bash \
    /usr/local/lmod/lmod/init/bash \
    /opt/apps/lmod/lmod/init/bash \
    /etc/profile.d/lmod.sh \
    /etc/profile.d/modules.sh \
    /usr/share/Modules/init/bash; do
    if [[ -f "$_init" ]]; then
        # shellcheck disable=SC1090
        source "$_init" && echo "Loaded module system: $_init" && break
    fi
done
set -u

# ---- Try to load a CUDA module -------------------------------------------
if command -v module &>/dev/null; then
    echo "Available modules (CUDA):"
    module avail 2>&1 | grep -i cuda || echo "  <none found>"
    for _cuda_mod in CUDA/12.4.0 CUDA/12.1.0 CUDA/12.0.0 CUDA/11.8.0 cuda/12.4 cuda/12.1 cuda; do
        if module load "$_cuda_mod" 2>/dev/null; then
            echo "Loaded module: $_cuda_mod"
            break
        fi
    done
else
    echo "module command not available"
fi

# ---- Locate CUDA_HOME if not already set ---------------------------------
if [[ -z "${CUDA_HOME:-}" ]]; then
    NVCC="$(which nvcc 2>/dev/null || true)"
    if [[ -n "$NVCC" ]]; then
        export CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
    else
        for _c in /usr/local/cuda /opt/cuda /usr/local/cuda-12.4 /usr/local/cuda-12.1 /usr/local/cuda-12.0 /usr/local/cuda-11.8; do
            if [[ -f "$_c/bin/nvcc" ]]; then
                export CUDA_HOME="$_c"; break
            fi
        done
    fi
fi

[[ -n "${CUDA_HOME:-}" ]] && export PATH="$CUDA_HOME/bin:$PATH" && export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "CUDA_HOME : ${CUDA_HOME:-<not found>}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true
echo ""

# ---- Install vLLM -----------------------------------------------------------
# Strategy: try multiple (vllm, torch, torchvision) version triples in order,
# for each CUDA tag, and VERIFY the import works before declaring success.
# This avoids the ABI-mismatch loop: "install succeeds but _C.abi3.so fails
# at runtime because vllm was compiled against a different torch version."
#
# Triples are listed newest-first.  Override the search list entirely by
# setting VLLM_VERSION / TORCH_VERSION / TV_VERSION env vars (skips the loop
# and uses only that combination).
#
# Format per entry: "vllm_ver:torch_ver:tv_ver"
# torchvision minor = torch_minor + 15  (torch 2.4 → tv 0.19, etc.)
# ---------------------------------------------------------------------------

echo "CUDA_HOME : ${CUDA_HOME:-<not set>}"
echo "nvcc      : $(which nvcc 2>/dev/null || echo '<not found>')"
echo ""

# If the caller pinned all three, use only that combo; otherwise auto-detect.
if [[ -n "${VLLM_VERSION:-}" && -n "${TORCH_VERSION:-}" && -n "${TV_VERSION:-}" ]]; then
    COMBOS=("${VLLM_VERSION}:${TORCH_VERSION}:${TV_VERSION}")
    echo "Using caller-specified versions: vllm=${VLLM_VERSION} torch=${TORCH_VERSION} tv=${TV_VERSION}"
else
    # Ordered newest → oldest; each triple is known to be ABI-compatible.
    COMBOS=(
        "0.8.5:2.6.0:0.21.0"
        "0.8.3:2.5.1:0.20.0"
        "0.7.3:2.5.1:0.20.0"
        "0.7.3:2.4.0:0.19.0"
        "0.6.1.post2:2.4.0:0.19.0"
    )
    echo "Will probe version combos: ${COMBOS[*]}"
fi
echo ""

SUCCESS=0
for COMBO in "${COMBOS[@]}"; do
    IFS=: read -r VLLM_VER TORCH_VER TV_VER <<< "${COMBO}"

    for CUDA_TAG in cu126 cu124 cu121; do
        echo "------------------------------------------------------------"
        echo "Trying vllm==${VLLM_VER}  torch==${TORCH_VER}+${CUDA_TAG}  tv==${TV_VER}+${CUDA_TAG}"

        # Install everything in one resolver pass so the version constraints
        # are visible to each other.  --extra-index-url makes both PyPI (for
        # vllm) and the PyTorch CUDA index (for torch/torchvision) available.
        if ! uv pip install \
                "torch==${TORCH_VER}+${CUDA_TAG}" \
                "torchvision==${TV_VER}+${CUDA_TAG}" \
                "vllm==${VLLM_VER}" \
                "setuptools>=77,<79" \
                --extra-index-url "https://download.pytorch.org/whl/${CUDA_TAG}" \
                2>/dev/null; then
            echo "  pip install failed — skipping"
            continue
        fi

        # Verify the C extension actually loads (catches ABI mismatches).
        IMPORT_OUT=$(uv run python -c "import vllm; print('OK', vllm.__version__)" 2>&1)
        if echo "${IMPORT_OUT}" | grep -q "^OK"; then
            echo "  Import check: ${IMPORT_OUT}"
            echo "  ✓ SUCCESS"
            SUCCESS=1
            CUDA_TAG_USED="${CUDA_TAG}"
            break 2
        else
            echo "  Import failed:"
            echo "${IMPORT_OUT}" | head -5 | sed 's/^/    /'
            echo "  Removing and trying next combo ..."
            # Wipe vllm + torch so the next iteration starts clean.
            uv pip uninstall vllm torch torchvision -y 2>/dev/null || true
        fi
    done
done

if [[ "${SUCCESS}" -eq 0 ]]; then
    echo ""
    echo "ERROR: No working vllm+torch combination found."
    echo "All tried combos produced an ABI mismatch at import time."
    echo ""
    echo "Hints:"
    echo "  1. Check GPU compute capability (nvidia-smi) — vllm >=0.7 dropped V100."
    echo "     For V100: VLLM_VERSION=0.6.1.post2 TORCH_VERSION=2.4.0 TV_VERSION=0.19.0 sbatch slurm/install_vllm.sh"
    echo "  2. Check what CUDA version the node has (nvcc --version)."
    echo "  3. Try cu121 only: edit COMBOS to remove cu126/cu124 entries."
    exit 1
fi

echo ""
TORCH_INFO=$(uv run python -c \
    "import torch; print(torch.__version__, '| CUDA:', torch.version.cuda)" \
    2>/dev/null || echo "unknown")
echo "torch : ${TORCH_INFO}"

echo ""
echo "Verifying install ..."
uv run python -c "import vllm; print(f'vllm {vllm.__version__} installed OK')"

echo "========================================"
echo "vLLM install complete."
echo "Finished : $(date)"
echo "========================================"
