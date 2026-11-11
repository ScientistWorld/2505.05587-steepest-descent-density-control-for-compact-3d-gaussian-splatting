#!/bin/bash
# Install Python packages into the container overlay (run by the system on a
# node with internet). Install into the SYSTEM Python; no virtual environments.
#
# Builds the two CUDA extensions of the official SteepGS / 3DGS code base
# (method/SteepGS/submodules: diff-gaussian-rasterization with splitting-matrix
# support, simple-knn). No GPU is needed at build time; kernels are compiled for
# the architectures in TORCH_CUDA_ARCH_LIST.

set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;8.9;9.0}"
export MAX_JOBS="${MAX_JOBS:-8}"

if ! command -v uv >/dev/null 2>&1; then
  python3 -m pip install --no-cache-dir uv
fi

uv pip install --system torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
uv pip install --system "numpy<2" pillow plyfile tqdm opencv-python-headless setuptools wheel ninja

# Locate the vendored extension sources (workspace copy), else fetch upstream.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC=""
for cand in "$HERE/../method/SteepGS/submodules" "/home/user/method/SteepGS/submodules"; do
  if [ -f "$cand/diff-gaussian-rasterization/setup.py" ] && [ -f "$cand/diff-gaussian-rasterization/third_party/glm/glm/glm.hpp" ]; then
    SRC="$(cd "$cand" && pwd)"
    break
  fi
done
if [ -z "$SRC" ]; then
  TMPSRC="$(mktemp -d)"
  git clone --depth 1 https://github.com/facebookresearch/SteepGS "$TMPSRC/SteepGS"
  git clone https://github.com/g-truc/glm.git "$TMPSRC/SteepGS/submodules/diff-gaussian-rasterization/third_party/glm"
  (cd "$TMPSRC/SteepGS/submodules/diff-gaussian-rasterization/third_party/glm" && git checkout 5c46b9c)
  sed -i 's|#include "cuda_runtime.h"|#include "cuda_runtime.h"\n#include <cfloat>|' "$TMPSRC/SteepGS/submodules/simple-knn/simple_knn.cu"
  SRC="$TMPSRC/SteepGS/submodules"
fi

# Build wheels from a scratch copy (no build artifacts in the workspace). The
# compile can exceed the 10-minute limit the setup wrapper puts on uv/pip
# calls, so the wheels are built with the interpreter's own pip/setuptools
# (`command` bypasses the wrapper functions) and then installed with uv.
BUILD="$(mktemp -d)"
cp -r "$SRC/diff-gaussian-rasterization" "$SRC/simple-knn" "$BUILD/"
rm -rf "$BUILD"/*/build "$BUILD"/*/*.egg-info
mkdir -p "$BUILD/wheels"
for ext in diff-gaussian-rasterization simple-knn; do
  if ! command python3 -m pip wheel --no-build-isolation --no-deps -w "$BUILD/wheels" "$BUILD/$ext"; then
    (cd "$BUILD/$ext" && command python3 setup.py bdist_wheel -d "$BUILD/wheels")
  fi
done
ls -la "$BUILD/wheels"
uv pip install --system --reinstall --no-deps "$BUILD"/wheels/*.whl
rm -rf "$BUILD"

python3 -c "import torch, torchvision, numpy, PIL, plyfile; print('setup ok', torch.__version__, numpy.__version__)"
python3 -c "import diff_gaussian_rasterization, simple_knn; print('extensions ok')" \
  || echo "WARNING: CUDA extensions installed but could not be imported here (no GPU driver?)"
