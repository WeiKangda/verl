#!/bin/bash
#SBATCH --job-name=fa-build
#SBATCH --account=157234358556
#SBATCH --partition=gpu_debug
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/user/%u/ExploratoryReasoning/verl/scripts/logs/fa_build_%j.out
#SBATCH --error=/scratch/user/%u/ExploratoryReasoning/verl/scripts/logs/fa_build_%j.err
#
# build_flash_attn.sbatch — compile flash-attn 2.8.1 from source
#
# Usage:
#   mkdir -p /scratch/user/$USER/ExploratoryReasoning/verl/scripts/logs
#   sbatch /scratch/user/$USER/ExploratoryReasoning/verl/scripts/build_flash_attn.sbatch
#
# Monitor:
#   squeue -u $USER
#   tail -f /scratch/user/$USER/ExploratoryReasoning/verl/scripts/logs/fa_build_<jobid>.out
#
# After success, the compiled wheel is archived to:
#   /scratch/group/p.cis240567.000/$USER/wheels/flash_attn-*.whl
# so you can reinstall it instantly next time without recompiling.

set -euo pipefail

FA_VERSION=2.8.1

echo "======================================"
echo "flash-attn source build"
echo "job id : $SLURM_JOB_ID"
echo "node   : $(hostname)"
echo "date   : $(date)"
echo "version: $FA_VERSION"
echo "======================================"

# ========== 1. Activate verl env ==========
source /scratch/user/$USER/ExploratoryReasoning/verl/scripts/activate_verl_hpc.sh

# ========== 2. Load a modern GCC (CUDA 12.8 needs GCC >= 11) ==========
module load GCC/13.2.0
echo "gcc: $(gcc --version | head -1)"
echo "nvcc: $(nvcc --version | tail -1)"
echo "nproc: $(nproc)"
free -g | head -2

# ========== 3. Clean previous build artifacts ==========
echo ""
echo "=== cleaning previous build artifacts ==="
uv pip uninstall flash-attn 2>&1 || true
rm -rf $UV_CACHE_DIR/sdists-v9/pypi/flash-attn
rm -rf $UV_CACHE_DIR/builds-v0
rm -rf $TMPDIR/tmpxft_* $TMPDIR/pip-* $TMPDIR/tmp* 2>/dev/null || true

# ========== 4. Build params ==========
# Force source build (skip prebuilt wheel which needs GLIBC 2.32+)
export FLASH_ATTENTION_FORCE_BUILD=TRUE

# Limit to H100 only. flash-attn 2.8+ hardcodes multiple archs in setup.py,
# so we set both the standard and the flash-attn-specific variable.
export TORCH_CUDA_ARCH_LIST="9.0"
export FLASH_ATTN_CUDA_ARCHS="90"

# Memory math: each nvcc job peaks at ~12 GB during cicc/ptxas.
# 4 parallel * 12 GB = 48 GB peak, well under the 256 GB limit.
# Going higher risks OOM kills on the heaviest kernels (hdim=256 bwd).
export MAX_JOBS=4
export NVCC_THREADS=1

echo ""
echo "=== build config ==="
echo "MAX_JOBS=$MAX_JOBS"
echo "NVCC_THREADS=$NVCC_THREADS"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "FLASH_ATTN_CUDA_ARCHS=$FLASH_ATTN_CUDA_ARCHS"
echo "FLASH_ATTENTION_FORCE_BUILD=$FLASH_ATTENTION_FORCE_BUILD"

# ========== 5. Background memory monitor ==========
MONITOR_LOG=/tmp/fa_mem_$SLURM_JOB_ID.log
(
    while true; do
        echo "[$(date '+%H:%M:%S')] $(free -g | awk '/^Mem/ {print "used="$3"G free="$4"G"}')" >> $MONITOR_LOG
        sleep 10
    done
) &
MONITOR_PID=$!
trap "kill $MONITOR_PID 2>/dev/null || true" EXIT

# ========== 6. Compile ==========
echo ""
echo "=== starting compile at $(date) ==="
time uv pip install flash-attn==$FA_VERSION --no-build-isolation 2>&1

BUILD_STATUS=$?
echo ""
echo "=== compile finished at $(date), exit=$BUILD_STATUS ==="

# ========== 7. Verify ==========
if [ $BUILD_STATUS -eq 0 ]; then
    echo ""
    echo "=== verification ==="
    python <<'EOF'
import torch
import flash_attn
print(f"flash_attn version: {flash_attn.__version__}")

from flash_attn import flash_attn_func
q = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16)
k = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16)
v = torch.randn(2, 8, 128, 64, device='cuda', dtype=torch.float16)
out = flash_attn_func(q, k, v)
print(f"flash_attn_func output shape: {out.shape}")
print("OK: flash-attn works on H100")
EOF
fi

# ========== 8. Archive the built wheel ==========
# Re-run pip wheel to produce a reusable .whl file (no recompile, it uses
# the already-built extension).
WHEEL_DIR=$VERL_ENV_ROOT/wheels
mkdir -p $WHEEL_DIR
echo ""
echo "=== archiving wheel to $WHEEL_DIR ==="
uv pip install build 2>&1 || true
python -m pip wheel --no-deps --no-build-isolation \
    flash-attn==$FA_VERSION -w $WHEEL_DIR 2>&1 || \
    echo "warning: could not archive wheel (you can still use the installed version)"

echo ""
echo "=== memory monitor summary (peak lines) ==="
sort -t'=' -k2 -rn $MONITOR_LOG 2>/dev/null | head -5 || echo "(no monitor data)"
rm -f $MONITOR_LOG

echo ""
echo "======================================"
echo "done at $(date)"
echo "======================================"
