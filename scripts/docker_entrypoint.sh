#!/bin/bash
# Docker entrypoint — activates venv, shows welcome message

export VIRTUAL_ENV=/workspace/video2robot/.venv
export PATH="/workspace/video2robot/.venv/bin:${PATH}"

PHMR_ROOT="/workspace/video2robot/third_party/PromptHMR"

echo ""
echo "┌─────────────────────────────────────────────────────────────┐"
echo "│  Hackathon Environment                                      │"
echo "│                                                             │"
echo "│  /workspace/video2robot/  — video2robot (venv: video2robot)│"
echo "│  /workspace/mjlab/        — mjlab RL (use: uv run ...)     │"
echo "└─────────────────────────────────────────────────────────────┘"

if [ ! -f "$PHMR_ROOT/data/body_models/smplx/SMPLX_NEUTRAL.npz" ] || \
   [ ! -f "$PHMR_ROOT/data/body_models/smpl/SMPL_NEUTRAL.pkl" ]; then
    echo ""
    echo "  ⚠  SMPL / SMPL-X body models not found."
    echo "     Run once (registration required):"
    echo "       bash /workspace/video2robot/scripts/fetch_body_models.sh"
    echo "     Register at: https://smpl-x.is.tue.mpg.de"
    echo "                  https://smpl.is.tue.mpg.de"
fi

echo ""

exec "$@"
