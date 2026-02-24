#!/bin/bash
# ============================================================================
# video2robot Installation Script (uv-based, no conda)
# ============================================================================
#
# Replaces the conda-based install_blackwell.sh with uv for use in containers
# or on any machine where conda is not available/desired.
#
# Prerequisites (system):
#   apt-get install -y gcc-11 g++-11 libsuitesparse-dev
#   uv must be installed: curl -LsSf https://astral.sh/uv/install.sh | sh
#
# Usage (run from video2robot repo root):
#   1. Extract data zip into third_party/PromptHMR/data/ BEFORE running this script:
#        unzip video2robot_data.zip -d third_party/PromptHMR/data/
#   2. Create venv and install PyTorch:
#        uv venv .venv --python 3.11
#        source .venv/bin/activate
#        UV_HTTP_TIMEOUT=300 uv pip install torch torchvision torchaudio xformers \
#          --index-url https://download.pytorch.org/whl/cu128
#   3. Run this script:
#        bash scripts/install_uv.sh
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
    echo "    unzip video2robot_data.zip -d third_party/PromptHMR/data/"
    exit 1
fi

# Verify we're inside a venv (not conda, not system Python)
if [ -z "$VIRTUAL_ENV" ]; then
    echo "ERROR: No active virtual environment detected."
    echo "  Please activate your uv venv first:"
    echo "    source .venv/bin/activate"
    exit 1
fi

# GCC-11 for CUDA extension builds
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export CUDAHOSTCXX=/usr/bin/g++-11

# ----------------------------------------------------------------------------
# Step 1: PyTorch check
# Already installed before running this script. Just verify.
# ----------------------------------------------------------------------------
echo ""
echo "[1/9] Verifying PyTorch + CUDA..."
python -c "
import torch
assert torch.cuda.is_available(), 'CUDA not available — check your PyTorch install'
print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name()}')
"

# ----------------------------------------------------------------------------
# Step 2: PromptHMR Python dependencies
# ----------------------------------------------------------------------------
echo ""
echo "[2/9] Installing PromptHMR Python dependencies..."
uv pip install -r "$PHMR_ROOT/requirements.txt"

# GMR dependencies
uv pip install -e "$REPO_ROOT/third_party/GMR"

# video2robot root package
uv pip install -e "$REPO_ROOT"

# ----------------------------------------------------------------------------
# Step 3: Submodules + Eigen
# ----------------------------------------------------------------------------
echo ""
echo "[3/9] Cloning submodules and Eigen..."

cd "$REPO_ROOT"
if [ -f ".gitmodules" ]; then
    git submodule update --init --recursive 2>/dev/null || true
fi

mkdir -p "$PHMR_ROOT/pipeline/droidcalib/thirdparty"
cd "$PHMR_ROOT/pipeline/droidcalib/thirdparty"

if [ ! -f "lietorch/lietorch/lietorch.py" ]; then
    rm -rf lietorch
    git clone --depth 1 https://github.com/princeton-vl/lietorch.git
fi

if [ ! -f "eigen/Eigen/Dense" ]; then
    rm -rf eigen
    git clone --depth 1 https://gitlab.com/libeigen/eigen.git
fi

# ----------------------------------------------------------------------------
# Step 4: Build lietorch CUDA extension
# ----------------------------------------------------------------------------
echo ""
echo "[4/9] Building lietorch..."
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
python setup.py install

# ----------------------------------------------------------------------------
# Step 5: Build droid_backends_intr CUDA extension
# ----------------------------------------------------------------------------
echo ""
echo "[5/9] Building droid_backends_intr..."

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
python setup.py install

# ----------------------------------------------------------------------------
# Step 6: torch_scatter + chumpy
# ----------------------------------------------------------------------------
echo ""
echo "[6/9] Installing torch_scatter and chumpy..."
cd "$PHMR_ROOT"

uv pip install torch_scatter --no-build-isolation

mkdir -p python_libs
if [ ! -f "python_libs/chumpy/setup.py" ]; then
    rm -rf python_libs/chumpy
    git clone https://github.com/Arthur151/chumpy python_libs/chumpy
fi
uv pip install -e python_libs/chumpy --no-build-isolation

# ----------------------------------------------------------------------------
# Step 7: detectron2 + sam2
# ----------------------------------------------------------------------------
echo ""
echo "[7/9] Installing detectron2 + sam2..."

uv pip install 'git+https://github.com/facebookresearch/detectron2.git' --no-build-isolation

# sam2: prefer custom wheel from the data zip (has load_video_frames_from_np patch)
if [ -f "data/wheels/sam2-1.5-cp311-cp311-linux_x86_64.whl" ]; then
    uv pip install data/wheels/sam2-1.5-cp311-cp311-linux_x86_64.whl
else
    echo "  WARNING: data/wheels/sam2-*.whl not found."
    echo "  Make sure you extracted the data zip before running this script."
    echo "  Falling back to PyPI sam2 — some video features may not work."
    uv pip install sam2
fi

# ----------------------------------------------------------------------------
# Step 8: Verify checkpoints (should all be present from the extracted zip)
# ----------------------------------------------------------------------------
echo ""
echo "[8/9] Verifying checkpoints from data zip..."
cd "$PHMR_ROOT"

MISSING=0
for f in \
    "data/pretrain/prompthmr" \
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
    echo ""
    echo "  WARNING: Some checkpoints are missing."
    echo "  Make sure you extracted the full data zip:"
    echo "    unzip video2robot_data.zip -d third_party/PromptHMR/data/"
else
    echo "  All checkpoints present: OK"
fi

# ----------------------------------------------------------------------------
# Step 9: SMPL / SMPL-X body models check + GMR symlink
# ----------------------------------------------------------------------------
echo ""
echo "[9/9] Checking body models and GMR symlink..."

if [ ! -d "$PHMR_ROOT/data/body_models/smplx" ] || \
   [ ! -f "$PHMR_ROOT/data/body_models/smpl/SMPL_NEUTRAL.pkl" ]; then
    echo ""
    echo "  ┌─────────────────────────────────────────────────────┐"
    echo "  │  Body models not found. Run after registration:     │"
    echo "  │    bash third_party/PromptHMR/scripts/fetch_smplx.sh│"
    echo "  │  Register at: https://smpl-x.is.tue.mpg.de          │"
    echo "  │               https://smpl.is.tue.mpg.de            │"
    echo "  └─────────────────────────────────────────────────────┘"
else
    echo "  Body models: OK"
fi

# GMR symlink: GMR expects smplx at assets/body_models/smplx/
GMR_DIR="$REPO_ROOT/third_party/GMR"
GMR_LINK="$GMR_DIR/assets/body_models/smplx"
PHMR_SMPLX="$PHMR_ROOT/data/body_models/smplx"

if [ -d "$GMR_DIR" ]; then
    mkdir -p "$GMR_DIR/assets/body_models"
    if [ ! -L "$GMR_LINK" ] && [ ! -d "$GMR_LINK" ]; then
        ln -s "$PHMR_SMPLX" "$GMR_LINK"
        echo "  GMR symlink created: $GMR_LINK -> $PHMR_SMPLX"
    else
        echo "  GMR symlink: already exists"
    fi
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
echo "Next: download body models (registration required)"
echo "  bash third_party/PromptHMR/scripts/fetch_smplx.sh"
echo "============================================"
