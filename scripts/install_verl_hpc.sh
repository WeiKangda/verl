#!/bin/bash
# install_verl_hpc.sh — FSDP + vLLM only, no Megatron, no SGLang
#
# Target: TAMU HPRC (or similar HPC with EasyBuild modules)
# Layout:
#   - Source code (this script + verl working copy):
#       /scratch/user/$USER/ExploratoryReasoning/verl
#   - Environment (venv + caches + HF downloads):
#       /scratch/group/p.cis240567.000/$USER
#     (group scratch, because user scratch inode count is nearly full)
#
# Usage:
#   1. salloc a GPU compute node (1 GPU is enough for install)
#   2. cd /scratch/user/$USER/ExploratoryReasoning/verl/scripts
#   3. bash install_verl_hpc.sh 2>&1 | tee install_$(date +%Y%m%d_%H%M).log

set -euo pipefail

# ========== 0. Path config ==========
# Env + caches live on the group scratch (has plenty of inodes).
export VERL_ENV_ROOT=/scratch/group/p.cis240567.000/$USER
# verl source code lives on user scratch alongside the research project.
export VERL_SRC=/scratch/user/$USER/ExploratoryReasoning/verl

mkdir -p $VERL_ENV_ROOT

export UV_CACHE_DIR=$VERL_ENV_ROOT/cache/uv
export UV_PYTHON_INSTALL_DIR=$VERL_ENV_ROOT/uv_python
export UV_TOOL_DIR=$VERL_ENV_ROOT/uv_tool
export UV_TOOL_BIN_DIR=$VERL_ENV_ROOT/bin
export XDG_BIN_HOME=$VERL_ENV_ROOT/bin
export XDG_DATA_HOME=$VERL_ENV_ROOT/xdg_data
export PIP_CACHE_DIR=$VERL_ENV_ROOT/cache/pip
export HF_HOME=$VERL_ENV_ROOT/cache/huggingface
export TORCH_HOME=$VERL_ENV_ROOT/cache/torch
export TRITON_CACHE_DIR=$VERL_ENV_ROOT/cache/triton
export XDG_CACHE_HOME=$VERL_ENV_ROOT/cache/xdg
export TMPDIR=$VERL_ENV_ROOT/tmp
mkdir -p $UV_CACHE_DIR $UV_PYTHON_INSTALL_DIR $UV_TOOL_DIR \
         $UV_TOOL_BIN_DIR $XDG_BIN_HOME $XDG_DATA_HOME \
         $PIP_CACHE_DIR $HF_HOME $TORCH_HOME \
         $TRITON_CACHE_DIR $XDG_CACHE_HOME $TMPDIR

# Sanity: make sure the verl source tree exists where we expect it.
if [ ! -d "$VERL_SRC" ]; then
    echo "ERROR: verl source not found at $VERL_SRC"
    echo "       Clone or rsync the ExploratoryReasoning repo there first."
    exit 1
fi

# ========== 1. HPC modules ==========
module purge
module load CUDA/12.8.0
module load WebProxy
export CUDA_HOME=${EBROOTCUDA:-/sw/eb/sw/CUDA/12.8.0}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
echo "=== CUDA check ==="
nvcc --version
nvidia-smi

# ========== 2. Install uv (into group scratch) ==========
if ! command -v uv &> /dev/null; then
    export UV_INSTALL_DIR=$VERL_ENV_ROOT/uv_bin
    mkdir -p $UV_INSTALL_DIR
    curl -LsSf https://astral.sh/uv/install.sh | \
        env UV_INSTALL_DIR=$UV_INSTALL_DIR INSTALLER_NO_MODIFY_PATH=1 sh
    export PATH=$UV_INSTALL_DIR:$PATH
fi
uv --version

# ========== 3. Create venv (Python 3.12, required by flash-attn wheel) ==========
cd $VERL_ENV_ROOT
uv python install 3.12
uv venv --python 3.12 $VERL_ENV_ROOT/verl_env
source $VERL_ENV_ROOT/verl_env/bin/activate
python --version

export MAX_JOBS=32

# ========== 4. Inference backend: vLLM only ==========
echo "[1/5] install vLLM (will pull torch 2.8 + cu12)"
uv pip install "vllm==0.11.0"

# ========== 5. Training basics ==========
echo "[2/5] install basic packages"
uv pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas "tensordict>=0.8.0,<=0.10.0,!=0.9.0" torchdata \
    "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 \
    liger-kernel mathruler pytest py-spy pre-commit ruff tensorboard

uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" \
    "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

# ========== 6. FlashAttention + FlashInfer ==========
echo "[3/5] install FlashAttention and FlashInfer"
# Try prebuilt wheel first; fall back to source build if GLIBC too old.
uv pip install flash-attn==2.8.1 || {
    echo "Prebuilt wheel failed (likely GLIBC mismatch), building from source..."
    export FLASH_ATTENTION_FORCE_BUILD=TRUE
    export TORCH_CUDA_ARCH_LIST="9.0"      # only H100, saves compile time
    export MAX_JOBS=4                       # keep memory under control (~50 GB peak)
    export NVCC_THREADS=1
    uv pip install flash-attn==2.8.1 --no-build-isolation
}

uv pip install flashinfer-python==0.3.1

# ========== 7. opencv fix ==========
echo "[4/5] fix opencv"
uv pip install opencv-python opencv-fixer
python -c "from opencv_fixer import AutoFix; AutoFix()" || echo "opencv-fixer warning (non-fatal)"

# ========== 8. Install verl itself (editable, from user scratch) ==========
echo "[5/5] install verl (editable) from $VERL_SRC"
cd $VERL_SRC
uv pip install --no-deps -e .

# ========== 9. Verification ==========
echo "=== verification ==="
python <<'EOF'
import torch, vllm, transformers
print(f"torch       : {torch.__version__}")
print(f"CUDA        : {torch.version.cuda}")
print(f"cuDNN       : {torch.backends.cudnn.version()}")
print(f"GPU         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
print(f"vllm        : {vllm.__version__}")
print(f"transformers: {transformers.__version__}")
import flash_attn; print(f"flash_attn  : {flash_attn.__version__}")
import verl; print(f"verl        : OK")
EOF

echo "=== Done ==="
