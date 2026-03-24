#!/bin/bash
# ---------------------------------------------------------------------------
# setup_cuda.sh — source this at the top of every GPU Slurm script.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/setup_cuda.sh"
#
# What it does:
#   1. Initialises Lmod / Environment Modules (if present on the node).
#   2. Loads the highest available CUDA module so that libcudnn.so.9 and
#      other CUDA libraries are on LD_LIBRARY_PATH before Python/torch start.
#   3. Sets CUDA_HOME from nvcc if not already set.
# ---------------------------------------------------------------------------

# Temporarily disable -u: Lmod init scripts reference unset variables.
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
        source "$_init" && break
    fi
done

set -u

if command -v module &>/dev/null; then
    for _cuda_mod in CUDA/12.6.0 CUDA/12.4.0 CUDA/12.1.0 CUDA/12.0.0 CUDA/11.8.0 \
                     cuda/12.6   cuda/12.4   cuda/12.1   cuda; do
        if module load "$_cuda_mod" 2>/dev/null; then
            echo "[setup_cuda] Loaded module: $_cuda_mod"
            break
        fi
    done
fi

# Fallback: set CUDA_HOME from nvcc if not already populated by the module.
if [[ -z "${CUDA_HOME:-}" ]]; then
    _nvcc="$(command -v nvcc 2>/dev/null || true)"
    if [[ -n "$_nvcc" ]]; then
        export CUDA_HOME="$(dirname "$(dirname "$_nvcc")")"
    else
        for _c in /usr/local/cuda /opt/cuda /usr/local/cuda-12.4 /usr/local/cuda-12.1 /usr/local/cuda-11.8; do
            if [[ -f "$_c/bin/nvcc" ]]; then
                export CUDA_HOME="$_c"; break
            fi
        done
    fi
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:${PATH:-}"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
