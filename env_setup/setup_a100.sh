#!/usr/bin/env bash
# Set up Contact-GraspNet for an A100 (sm_80) in a project-local conda env.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT="$ROOT/CGN_3D_TF213"
ENV_PREFIX="${ENV_PREFIX:-$(conda info --base)/envs/cgn_3d}"
CUDA_HOME="${CUDA_HOME:-/opt/cuda/12.4}"

if [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "CUDA 12.4 was not found at $CUDA_HOME." >&2
    echo "Run 'module load cuda/12.4', then rerun with CUDA_HOME=/opt/cuda/12.4." >&2
    exit 1
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    conda create --prefix "$ENV_PREFIX" python=3.9 pip -y
fi

"$ENV_PREFIX/bin/python" -m pip install --upgrade pip
"$ENV_PREFIX/bin/python" -m pip install -r "$PROJECT/env_setup/requirements-a100.txt"

export CUDA_HOME
export PATH="$ENV_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
cd "$PROJECT/env_setup"
bash ./recompile_ops.sh

"$ENV_PREFIX/bin/python" -c 'import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices("GPU"))'
