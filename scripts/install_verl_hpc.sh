#!/bin/bash
# install_verl_hpc.sh — FSDP + vLLM only, no Megatron, no SGLang
#
# Pinned to match verl v0.5.x reference install
# (scripts/install_vllm_sglang_mcore.sh): torch 2.6 + cu124 + vllm 0.8.5.post1
# + flash-attn 2.7.4.post1 + flashinfer 0.2.2.post1, Python 3.10.
#
# Note: v0.6.x async rollout code requires vllm 0.9.0, which pulls in
# torch 2.7 and breaks flash-attn <=2.7.x. v0.5.x is fully compatible
# with this stack.
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
# Reference wheels are built against cu124 (CUDA 12.4). Pinned to CUDA/12.4.0
# on TAMU HPRC so runtime matches the wheels exactly.
module purge
module load CUDA/12.4.0
module load WebProxy
export CUDA_HOME=${EBROOTCUDA:?EBROOTCUDA not set after module load CUDA/12.4.0}
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
echo "=== CUDA module: CUDA/12.4.0 ==="
echo "CUDA_HOME = $CUDA_HOME"
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

# ========== 3. Create venv (Python 3.10, required by reference flash-attn wheel) ==========
cd $VERL_ENV_ROOT
uv python install 3.10
uv venv --python 3.10 $VERL_ENV_ROOT/verl_env
source $VERL_ENV_ROOT/verl_env/bin/activate
python --version

export MAX_JOBS=32

# ========== 4. Inference backend: vLLM only ==========
# Pinned versions from verl v0.6.x install_vllm_sglang_mcore.sh.
echo "[1/5] install vLLM + torch 2.6 (cu124)"
uv pip install --no-cache-dir \
    "vllm==0.8.5.post1" \
    "torch==2.6.0" \
    "torchvision==0.21.0" \
    "torchaudio==2.6.0" \
    "tensordict==0.6.2" \
    torchdata

# ========== 5. Training basics ==========
echo "[2/5] install basic packages"
uv pip install "transformers[hf_xet]>=4.51.0,<5.0.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 \
    liger-kernel mathruler pytest py-spy pyext pre-commit ruff tensorboard

uv pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" \
    "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

# ========== 6. FlashAttention + FlashInfer ==========
# Prebuilt wheels for torch 2.6 + cu124 + Python 3.10 (cxx11abi=FALSE).
echo "[3/5] install FlashAttention and FlashInfer"
FA_WHL=flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
FI_WHL=flashinfer_python-0.2.2.post1+cu124torch2.6-cp38-abi3-linux_x86_64.whl

cd $TMPDIR
wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/$FA_WHL
uv pip install --no-cache-dir "$TMPDIR/$FA_WHL"

wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.2.post1/$FI_WHL
uv pip install --no-cache-dir "$TMPDIR/$FI_WHL"

# ========== 6b. Patch activate script: torch/lib on LD_LIBRARY_PATH ==========
# HPC nodes often strip RPATH from .so files, so flash-attn (and other torch
# C++ extensions) can't find libc10.so at runtime.  Baking the path into the
# activate script means every future `source .../activate` just works.
TORCH_LIB="$VERL_ENV_ROOT/verl_env/lib/python3.10/site-packages/torch/lib"
if ! grep -q "torch/lib" "$VERL_ENV_ROOT/verl_env/bin/activate" 2>/dev/null; then
    cat >> "$VERL_ENV_ROOT/verl_env/bin/activate" <<ACTIVATE_EOF

# flash-attn needs torch/lib on LD_LIBRARY_PATH (HPC RPATH workaround)
export LD_LIBRARY_PATH="$TORCH_LIB:\${LD_LIBRARY_PATH}"
ACTIVATE_EOF
    echo "  -> patched activate script with torch/lib LD_LIBRARY_PATH"
fi

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
import flashinfer; print(f"flashinfer  : {flashinfer.__version__}")
import verl; print(f"verl        : OK")
EOF

echo "=== Done ==="
