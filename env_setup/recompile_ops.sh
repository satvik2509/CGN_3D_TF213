#!/usr/bin/env bash
set -euo pipefail

# TensorFlow 2.15 uses CUDA 12.x. This cluster provides CUDA 12.4.
export CUDA_HOME="${CUDA_HOME:-/opt/cuda/12.4}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# CUDA 12.4 supports host GCC versions through 13.  The system default is
# GCC 15, so explicitly select the installed GCC 13 toolchain for both nvcc
# device compilation and host-side shared-library linking.
export CC="${CC:-gcc-13}"
export CXX="${CXX:-g++-13}"
if ! command -v "$CC" >/dev/null || ! command -v "$CXX" >/dev/null; then
    echo "CUDA 12.4 requires GCC/G++ <= 13; install or load gcc-13." >&2
    exit 1
fi

CUDA_INCLUDE="-I$CUDA_HOME/include"
CUDA_LIB="-L$CUDA_HOME/lib64"

# Ubuntu's glibc 2.41 exposes C23 cospi/sinpi/rsqrt declarations whenever
# _GNU_SOURCE is enabled. CUDA 12.4 declares the same device functions with
# incompatible exception specifications. Compile the legacy PointNet++ ops
# against the POSIX feature set to avoid those host-only declarations, while
# explicitly including alloca.h because Eigen requires alloca().
GLIBC_COMPAT_FLAGS=(-U_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -include alloca.h)

# Get TF flags from the contact_graspnet_env
TF_VERSION=$(python -c 'import tensorflow as tf; print(tf.__version__)')
if [[ "$TF_VERSION" != 2.15.* ]]; then
    echo "Expected TensorFlow 2.15.x, found $TF_VERSION" >&2
    exit 1
fi
read -r -a TF_CFLAGS <<< "$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')"
read -r -a TF_LFLAGS <<< "$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')"

echo "TensorFlow: $TF_VERSION"
echo "Host compiler: $($CXX --version | head -n1)"
echo "TF_CFLAGS: ${TF_CFLAGS[*]}"
echo "TF_LFLAGS: ${TF_LFLAGS[*]}"
echo ""

# GPU architectures: sm_80 (A100), sm_86 (RTX 30xx), sm_90 (H100/H200)
ARCH_FLAGS=(
    --generate-code=arch=compute_80,code=sm_80
    --generate-code=arch=compute_86,code=sm_86
    --generate-code=arch=compute_90,code=sm_90
    --generate-code=arch=compute_90,code=compute_90
)

BASE="$(dirname "$(realpath "$0")")/.."

echo "=== Compiling sampling ==="
cd "$BASE/pointnet2/tf_ops/sampling"
nvcc -ccbin "$CXX" -std=c++17 -c -o tf_sampling_g.cu.o tf_sampling_g.cu \
    $CUDA_INCLUDE "${TF_CFLAGS[@]}" -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
    "${GLIBC_COMPAT_FLAGS[@]}" "${ARCH_FLAGS[@]}"
"$CXX" -std=c++17 -shared -o tf_sampling_so.so tf_sampling.cpp \
    tf_sampling_g.cu.o $CUDA_INCLUDE "${TF_CFLAGS[@]}" -fPIC -lcudart "${TF_LFLAGS[@]}" $CUDA_LIB
echo "sampling: OK"

echo "=== Compiling grouping ==="
cd "$BASE/pointnet2/tf_ops/grouping"
nvcc -ccbin "$CXX" -std=c++17 -c -o tf_grouping_g.cu.o tf_grouping_g.cu \
    $CUDA_INCLUDE "${TF_CFLAGS[@]}" -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC \
    "${GLIBC_COMPAT_FLAGS[@]}" "${ARCH_FLAGS[@]}"
"$CXX" -std=c++17 -shared -o tf_grouping_so.so tf_grouping.cpp \
    tf_grouping_g.cu.o $CUDA_INCLUDE "${TF_CFLAGS[@]}" -fPIC -lcudart "${TF_LFLAGS[@]}" $CUDA_LIB
echo "grouping: OK"

echo "=== Compiling 3d_interpolation ==="
cd "$BASE/pointnet2/tf_ops/3d_interpolation"
"$CXX" -std=c++17 tf_interpolate.cpp -o tf_interpolate_so.so \
    -shared -fPIC "${TF_CFLAGS[@]}" "${TF_LFLAGS[@]}" -O2
echo "interpolation: OK"

echo ""
echo "=== All ops recompiled successfully! ==="
HAS_GPU=$(python -c 'import tensorflow as tf; print(1 if tf.config.list_physical_devices("GPU") else 0)')
if [[ "$HAS_GPU" == "1" ]]; then
    echo "Testing sampling..."
    cd "$BASE/pointnet2/tf_ops/sampling" && python tf_sampling.py
    echo "Testing grouping..."
    cd "$BASE/pointnet2/tf_ops/grouping" && python tf_grouping_op_test.py
    echo "Testing interpolation..."
    cd "$BASE/pointnet2/tf_ops/3d_interpolation" && python tf_interpolate_op_test.py
    echo "=== All tests passed! ==="
else
    echo "No TensorFlow GPU detected; skipping GPU-only sampling/grouping op tests."
    echo "Testing CPU interpolation op..."
    cd "$BASE/pointnet2/tf_ops/3d_interpolation" && python tf_interpolate_op_test.py
    echo "=== CPU-compatible op tests passed; run GPU tests once /dev/nvidia* is available. ==="
fi
