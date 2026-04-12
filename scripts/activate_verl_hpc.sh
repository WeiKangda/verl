#!/bin/bash
# activate_verl_hpc.sh — source this on the HPC to activate the verl env
#
# Layout:
#   - Env + caches: /scratch/group/p.cis240567.000/$USER  (VERL_ENV_ROOT)
#   - verl source : /scratch/user/$USER/ExploratoryReasoning/verl  (VERL_SRC)
#
# CUDA is pinned to CUDA/12.4.0 to match the cu124 wheels used at install time
# (torch 2.6 + flash-attn 2.7.4.post1 + flashinfer 0.2.2.post1).
#
# Usage:
#   source /scratch/user/$USER/ExploratoryReasoning/verl/scripts/activate_verl_hpc.sh

export VERL_ENV_ROOT=/scratch/group/p.cis240567.000/$USER
export VERL_SRC=/scratch/user/$USER/ExploratoryReasoning/verl

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
export PATH=$VERL_ENV_ROOT/uv_bin:$PATH

module purge
module load CUDA/12.4.0
module load WebProxy
export CUDA_HOME=$EBROOTCUDA
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

source $VERL_ENV_ROOT/verl_env/bin/activate
echo "verl env activated: $(which python)"
echo "  VERL_ENV_ROOT = $VERL_ENV_ROOT"
echo "  VERL_SRC      = $VERL_SRC"
echo "  CUDA module   = CUDA/12.4.0 ($CUDA_HOME)"
