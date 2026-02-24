# ============================================================================
# video2robot + mjlab Docker Image
# ============================================================================
#
# Target: NVIDIA H100 / B200 (sm_90, sm_100) on Nebius Cloud
# Base:   CUDA 12.8 devel + Ubuntu 22.04
#
# Layout inside container:
#   /workspace/video2robot/   — video2robot repo, venv at .venv (prompt: video2robot)
#   /workspace/mjlab/         — mjlab repo, venv at .venv (prompt: mjlab, use uv run)
#
# Build:
#   docker build \
#     --build-arg DATA_ZIP_ID=1nR10gHr0MUIziZnkolb07w8RtnNAuljv \
#     -t cr.nebius.cloud/YOUR_REGISTRY/video2robot:latest .
#
# Run:
#   docker run --gpus all -it \
#     -v video2robot-data:/workspace/video2robot/third_party/PromptHMR/data/body_models \
#     cr.nebius.cloud/YOUR_REGISTRY/video2robot:latest
#
# First run inside container (one time):
#   bash /workspace/video2robot/fetch_body_models.sh
#
# mjlab usage:
#   cd /workspace/mjlab
#   uv run train Mjlab-Velocity-Flat-Unitree-G1 --env.scene.num-envs 4096
#
# ============================================================================

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# ----------------------------------------------------------------------------
# System dependencies
# gcc-12 installed and symlinked as gcc-11 — some packages (e.g. pyliblzfse)
# hardcode gcc-11 in their build system regardless of CC env var
# ninja-build dramatically speeds up CUDA extension compilation
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y \
    gcc-12 g++-12 \
    git wget curl unzip \
    libsuitesparse-dev \
    ninja-build \
    python3-dev \
    ca-certificates \
    && ln -s /usr/bin/gcc-12 /usr/bin/gcc-11 \
    && ln -s /usr/bin/g++-12 /usr/bin/g++-11 \
    && rm -rf /var/lib/apt/lists/*

# CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# ----------------------------------------------------------------------------
# Install uv
# ----------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
ENV UV_HTTP_TIMEOUT=300

# ----------------------------------------------------------------------------
# Clone repo
# ----------------------------------------------------------------------------
WORKDIR /workspace
RUN git clone --recursive https://github.com/DavidDobas/video2robot-hackathon.git video2robot
WORKDIR /workspace/video2robot

# Remove git remote so participants can't accidentally push
RUN git remote remove origin

# ----------------------------------------------------------------------------
# Create venv + install PyTorch (big layer, cache separately)
# Prompt set to "video2robot" so shell shows (video2robot) when activated
# ----------------------------------------------------------------------------
RUN uv venv .venv --python 3.11 --prompt video2robot
ENV VIRTUAL_ENV=/workspace/video2robot/.venv
ENV PATH="/workspace/video2robot/.venv/bin:${PATH}"

RUN uv pip install torch==2.6.0 torchvision torchaudio xformers \
    --index-url https://download.pytorch.org/whl/cu128

# pip must be present in the venv — chumpy's setup.py imports it internally
RUN uv pip install pip

# ----------------------------------------------------------------------------
# Download and extract data zip (checkpoints, wheels, annotations, examples)
# Excludes body_models (licensed, downloaded at runtime)
# ----------------------------------------------------------------------------
ARG DATA_ZIP_ID=1nR10gHr0MUIziZnkolb07w8RtnNAuljv

RUN uv pip install gdown && \
    gdown ${DATA_ZIP_ID} -O /tmp/video2robot_data.zip && \
    unzip -o /tmp/video2robot_data.zip -d third_party/PromptHMR/data/ && \
    rm /tmp/video2robot_data.zip

# ----------------------------------------------------------------------------
# Run install script (builds CUDA extensions, installs all Python deps)
# ----------------------------------------------------------------------------
RUN bash scripts/install_uv.sh

# ----------------------------------------------------------------------------
# Clone mjlab (uses its own uv-managed venv via uv run, prompt: mjlab)
# ----------------------------------------------------------------------------
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/mujocolab/mjlab.git mjlab && \
    cd mjlab && \
    git remote remove origin

# Pre-install mjlab deps so first `uv run` is instant (no sync delay)
WORKDIR /workspace/mjlab
RUN uv sync --frozen

# ----------------------------------------------------------------------------
# Entrypoint — drop to bash, remind user to run fetch_body_models.sh
# ----------------------------------------------------------------------------
COPY scripts/docker_entrypoint.sh /docker_entrypoint.sh
RUN chmod +x /docker_entrypoint.sh

WORKDIR /workspace

ENTRYPOINT ["/docker_entrypoint.sh"]
CMD ["bash"]
