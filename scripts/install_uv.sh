#!/bin/bash
# ============================================================================
# video2robot Installation Script (uv-based, no conda)
# ============================================================================
#
# Replaces the conda-based install_blackwell.sh with uv for use in containers
# or on any machine where conda is not available/desired.
#
# Prerequisites (system):
#   apt-get install -y gcc-12 g++-12 libsuitesparse-dev ninja-build
#   ln -s /usr/bin/gcc-12 /usr/bin/gcc-11  # if gcc-11 not available
#   ln -s /usr/bin/g++-12 /usr/bin/g++-11
#   uv must be installed: curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Usage (run from video2robot repo root):
#   1. Extract data zip into third_party/PromptHMR/data/ BEFORE running this script:
#        unzip -o video2robot_data.zip -d third_party/PromptHMR/data/
#   2. Create venv and install PyTorch:
#        uv venv .venv --python 3.11 --prompt video2robot
#        source .venv/bin/activate
#        UV_HTTP_TIMEOUT=300 uv pip install torch==2.6.0 torchvision torchaudio xformers \
#          --index-url https://download.pytorch.org/whl/cu128
#        uv pip install pip  # needed by chumpy's setup.py
#   3. Run this script:
#        bash scripts/install_uv.sh
#
# Idempotent: steps that are already complete are skipped automatically.
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PHMR_ROOT="$REPO_ROOT/third_party/PromptHMR"

cd "$PHMR_ROOT"

echo "============================================"
echo "video2robot uv Installation"
echo "  REPO:  $REPO_ROOT"
echo "  PHMR:  $PHMR_ROOT"
echo "  Python: $(python --version)"
echo "============================================"

# Verify data zip has been extracted (checkpoints must be present before running)
if [ ! -d "$PHMR_ROOT/data/pretrain" ]; then
    echo "ERROR: third_party/PromptHMR/data/pretrain not found."
    echo "  Please extract the data zip first:"
    echo "    unzip -o video2robot_data.zip -d third_party/PromptHMR/data/"
    exit 1
fi

# Verify we're inside a venv (not conda, not system Python)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: No active virtual environment detected."
    echo "  Please activate your uv venv first:"
    echo "    source .venv/bin/activate"
    exit 1
fi

# GCC compiler setup — prefer gcc-11, fall back to gcc-12, then gcc-13
for ver in 11 12 13; do
    if [ -f "/usr/bin/gcc-$ver" ]; then
        export CC=/usr/bin/gcc-$ver
        export CXX=/usr/bin/g++-$ver
        export CUDAHOSTCXX=/usr/bin/g++-$ver
        echo "  Compiler: gcc-$ver"
        break
    fi
done

if [ -z "$CC" ]; then
    echo "ERROR: No suitable gcc found (tried gcc-11, gcc-12, gcc-13)"
    exit 1
fi

# Auto-detect CUDA_HOME if not set
if [ -z "$CUDA_HOME" ]; then
    for candidate in \
        /usr/local/cuda \
        /usr/lib/nvidia-cuda-toolkit \
        /usr/local/cuda-12.8 \
        /usr/local/cuda-12.6 \
        /usr/local/cuda-12.4 \
        /usr/local/cuda-12.1 \
        /usr/local/cuda-12.0
    do
        if [ -f "$candidate/bin/nvcc" ]; then
            export CUDA_HOME="$candidate"
            break
        fi
    done
fi

if [ -z "$CUDA_HOME" ]; then
    echo "ERROR: Could not find CUDA installation. Set CUDA_HOME manually:"
    echo "  export CUDA_HOME=/path/to/cuda"
    exit 1
fi
echo "  CUDA_HOME: $CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# On Ubuntu apt installs, cuda.h may live in /usr/lib/cuda/include separately
if [ -f "/usr/lib/cuda/include/cuda.h" ] && [ "$CUDA_HOME" != "/usr/lib/cuda" ]; then
    export CPATH="/usr/lib/cuda/include:${CPATH:-}"
    echo "  CPATH patched: /usr/lib/cuda/include (Ubuntu split toolkit)"
fi

# Helper: check if a Python module is importable
can_import() { python -c "import $1" 2>/dev/null; }

# ----------------------------------------------------------------------------
# Step 1: PyTorch check
# ----------------------------------------------------------------------------
echo ""
echo "[1/8] Verifying PyTorch + CUDA..."
if [ -n "$SKIP_CUDA_CHECK" ]; then
    echo "  SKIP_CUDA_CHECK set — skipping GPU assertion (build-time mode)"
    python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA toolkit {torch.version.cuda} (GPU not checked)')"
else
    python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check your PyTorch install'
print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name()}')
"
fi

# Derive PyTorch version string for PyG wheel index (e.g. "2.6.0+cu124")
TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
CUDA_TAG=$(python -c "import torch; v=torch.version.cuda.replace('.',''); print(f'cu{v}')")

# Pin torch so no subsequent install can silently upgrade it
TORCH_PIN=$(python -c "import torch; v=torch.__version__.split('+')[0]; print(f'torch=={v}')")
TORCH_CONSTRAINTS="$(mktemp /tmp/torch-constraints-XXXX.txt)"
echo "$TORCH_PIN" > "$TORCH_CONSTRAINTS"
echo "  Pinned: $TORCH_PIN (constraints: $TORCH_CONSTRAINTS)"

# ----------------------------------------------------------------------------
# Step 2: PromptHMR + GMR + video2robot Python dependencies
# uv pip install is fast and idempotent — always run to catch any missing deps
# ----------------------------------------------------------------------------
echo ""
echo "[2/8] Installing Python dependencies..."
uv pip install --constraint "$TORCH_CONSTRAINTS" -r "$PHMR_ROOT/requirements.txt"
uv pip install --constraint "$TORCH_CONSTRAINTS" -e "$REPO_ROOT/third_party/GMR"
# GMR requires smplx from git HEAD which defaults to num_betas=16, breaking PromptHMR
# checkpoints (trained with 10 betas). Force PyPI 0.1.28 which keeps num_betas=10.
uv pip install --constraint "$TORCH_CONSTRAINTS" --reinstall smplx==0.1.28
uv pip install --constraint "$TORCH_CONSTRAINTS" -e "$REPO_ROOT"

# ----------------------------------------------------------------------------
# Step 3: Submodules + Eigen (idempotent by directory checks)
# ----------------------------------------------------------------------------
echo ""
echo "[3/8] Cloning submodules and Eigen..."

cd "$REPO_ROOT"
if [ -f ".gitmodules" ]; then
    git submodule update --init --recursive 2>/dev/null || true
fi

mkdir -p "$PHMR_ROOT/pipeline/droidcalib/thirdparty"
cd "$PHMR_ROOT/pipeline/droidcalib/thirdparty"

if [ ! -f "lietorch/lietorch/lietorch.py" ]; then
    rm -rf lietorch
    git clone --depth 1 https://github.com/princeton-vl/lietorch.git
else
    echo "  lietorch source: already present"
fi

if [ ! -f "eigen/Eigen/Dense" ]; then
    rm -rf eigen
    git clone --depth 1 https://gitlab.com/libeigen/eigen.git
else
    echo "  eigen: already present"
fi

# ----------------------------------------------------------------------------
# Step 4: Build lietorch CUDA extension
# ----------------------------------------------------------------------------
echo ""
if can_import lietorch; then
    echo "[4/8] lietorch: already installed, skipping build"
else
    echo "[4/8] Building lietorch..."
    cd "$PHMR_ROOT/pipeline/droidcalib"

    cat > setup.py << 'EOF'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='lietorch',
    version='0.3',
    packages=['lietorch'],
    package_dir={'': 'thirdparty/lietorch'},
    ext_modules=[
        CUDAExtension('lietorch_backends',
            include_dirs=[
                osp.join(ROOT, 'thirdparty/lietorch/lietorch/include'),
                osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'thirdparty/lietorch/lietorch/src/lietorch.cpp',
                'thirdparty/lietorch/lietorch/src/lietorch_gpu.cu',
                'thirdparty/lietorch/lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={
                'cxx': ['-O2'],
                'nvcc': ['-O2',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_90,code=sm_90',
                    '-gencode=arch=compute_100,code=sm_100',
                    '-gencode=arch=compute_120,code=sm_120',
                ]}),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
EOF

    rm -rf build/
    pip install --no-build-isolation .
fi

# ----------------------------------------------------------------------------
# Step 5: Build droid_backends_intr CUDA extension
# ----------------------------------------------------------------------------
echo ""
if can_import droid_backends_intr; then
    echo "[5/8] droid_backends_intr: already installed, skipping build"
else
    echo "[5/8] Building droid_backends_intr..."
    cd "$PHMR_ROOT/pipeline/droidcalib"

    # PyTorch 2.9 API compatibility
    sed -i 's/fmap1\.type()/fmap1.scalar_type()/g' src/altcorr_kernel.cu 2>/dev/null || true
    sed -i 's/volume\.type()/volume.scalar_type()/g' src/correlation_kernels.cu 2>/dev/null || true

    cat > setup.py << 'EOF'
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os.path as osp
ROOT = osp.dirname(osp.abspath(__file__))

setup(
    name='droid_backends_intr',
    version='0.3',
    ext_modules=[
        CUDAExtension('droid_backends_intr',
            include_dirs=[osp.join(ROOT, 'thirdparty/eigen')],
            sources=[
                'src/droid.cpp',
                'src/droid_kernels.cu',
                'src/correlation_kernels.cu',
                'src/altcorr_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3',
                    '-gencode=arch=compute_75,code=sm_75',
                    '-gencode=arch=compute_80,code=sm_80',
                    '-gencode=arch=compute_86,code=sm_86',
                    '-gencode=arch=compute_90,code=sm_90',
                    '-gencode=arch=compute_100,code=sm_100',
                    '-gencode=arch=compute_120,code=sm_120',
                ]
            }),
    ],
    cmdclass={ 'build_ext' : BuildExtension }
)
EOF

    rm -rf build/
    pip install --no-build-isolation .
fi

# ----------------------------------------------------------------------------
# Step 6: torch_scatter + chumpy
# torch_scatter: use PyG pre-built wheel (much faster than building from source)
# ----------------------------------------------------------------------------
echo ""
cd "$PHMR_ROOT"

if can_import torch_scatter && can_import chumpy; then
    echo "[6/8] torch_scatter + chumpy: already installed, skipping"
else
    echo "[6/8] Installing torch_scatter and chumpy..."

    if ! can_import torch_scatter; then
        echo "  Installing torch_scatter from PyG pre-built wheel..."
        uv pip install --constraint "$TORCH_CONSTRAINTS" torch-scatter \
            -f "https://data.pyg.org/whl/torch-${TORCH_VERSION}.html" \
            || uv pip install --constraint "$TORCH_CONSTRAINTS" torch_scatter --no-build-isolation
    fi

    if ! can_import chumpy; then
        mkdir -p python_libs
        if [ ! -f "python_libs/chumpy/setup.py" ]; then
            rm -rf python_libs/chumpy
            git clone https://github.com/Arthur151/chumpy python_libs/chumpy
        fi
        uv pip install --constraint "$TORCH_CONSTRAINTS" -e python_libs/chumpy --no-build-isolation
    fi
fi

# ----------------------------------------------------------------------------
# Step 7: detectron2 + sam2
# ----------------------------------------------------------------------------
echo ""

if can_import detectron2 && can_import sam2; then
    echo "[7/8] detectron2 + sam2: already installed, skipping"
else
    echo "[7/8] Installing detectron2 + sam2..."

    if ! can_import detectron2; then
        uv pip install --constraint "$TORCH_CONSTRAINTS" 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation
    fi

    if ! can_import sam2; then
        if [ -f "data/wheels/sam2-1.5-cp311-cp311-linux_x86_64.whl" ]; then
            uv pip install --constraint "$TORCH_CONSTRAINTS" data/wheels/sam2-1.5-cp311-cp311-linux_x86_64.whl
        else
            echo "  WARNING: data/wheels/sam2-*.whl not found, installing from PyPI..."
            uv pip install --constraint "$TORCH_CONSTRAINTS" sam2
        fi
    fi
fi

# ----------------------------------------------------------------------------
# Step 8: Verify checkpoints
# ----------------------------------------------------------------------------
echo ""
echo "[8/8] Verifying checkpoints from data zip..."
cd "$PHMR_ROOT"

MISSING=0
for f in \
    "data/pretrain/phmr" \
    "data/pretrain/sam2_ckpts/sam2_hiera_tiny.pt" \
    "data/pretrain/sam2_ckpts/keypoint_rcnn_5ad38f.pkl" \
    "data/pretrain/vitpose-h-coco_25.pth" \
    "data/pretrain/droidcalib.pth" \
    "data/pretrain/camcalib_sa_biased_l2.ckpt"
do
    if [ ! -e "$PHMR_ROOT/$f" ]; then
        echo "  MISSING: $f"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 1 ]; then
    echo "  WARNING: Some checkpoints are missing."
    echo "  Make sure you extracted the full data zip:"
    echo "    unzip -o video2robot_data.zip -d third_party/PromptHMR/data/"
else
    echo "  All checkpoints present: OK"
fi

# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------
echo ""
echo "============================================"
echo "Verifying installation..."
echo "============================================"

python -c "
import torch
print(f'PyTorch:    {torch.__version__}')
print(f'CUDA:       {torch.cuda.is_available()}')
print(f'GPU:        {torch.cuda.get_device_name() if torch.cuda.is_available() else \"N/A\"}')

import xformers
print(f'xformers:   {xformers.__version__}')

import lietorch
print('lietorch:   OK')

import droid_backends_intr
print('droid_backends_intr: OK')

import chumpy
print('chumpy:     OK')

import detectron2
print(f'detectron2: {detectron2.__version__}')

import sam2
print('sam2:       OK')

print()
print('All done!')
"

echo ""
echo "============================================"
echo "Installation complete!"
echo "============================================"
