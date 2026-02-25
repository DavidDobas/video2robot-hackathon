#!/bin/bash
# ============================================================================
# Download SMPL / SMPL-X body models (registration required)
# ============================================================================
#
# Register at:
#   https://smpl-x.is.tue.mpg.de
#   https://smpl.is.tue.mpg.de
#
# Usage (from repo root):
#   bash scripts/fetch_body_models.sh
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PHMR_ROOT="$REPO_ROOT/third_party/PromptHMR"
GMR_ROOT="$REPO_ROOT/third_party/GMR"

urle() {
    [[ "${1}" ]] || return 1
    local LANG=C i x
    for (( i = 0; i < ${#1}; i++ )); do
        x="${1:i:1}"
        [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"
    done
    echo
}

if [ -f "$PHMR_ROOT/data/body_models/smplx/SMPLX_NEUTRAL.npz" ] && \
   [ -f "$PHMR_ROOT/data/body_models/smpl/SMPL_NEUTRAL.pkl" ]; then
    echo "Body models already present. Nothing to do."
    exit 0
fi

mkdir -p "$PHMR_ROOT/data/body_models"

echo ""
echo "Register at: https://smpl-x.is.tue.mpg.de"
read -p "SMPL-X username: " smplx_user
read -sp "SMPL-X password: " smplx_pass && echo
smplx_user=$(urle "$smplx_user")
smplx_pass=$(urle "$smplx_pass")

wget --post-data "username=$smplx_user&password=$smplx_pass" \
    'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=models_smplx_v1_1.zip' \
    -O /tmp/smplx.zip --no-check-certificate --continue
mkdir -p "$PHMR_ROOT/data/body_models/smplx"
unzip -o /tmp/smplx.zip -d "$PHMR_ROOT/data/body_models/smplx"
mv "$PHMR_ROOT/data/body_models/smplx/models/smplx/"* "$PHMR_ROOT/data/body_models/smplx/"
rm -rf "$PHMR_ROOT/data/body_models/smplx/models" /tmp/smplx.zip

echo ""
echo "Register at: https://smpl.is.tue.mpg.de"
read -p "SMPL username: " smpl_user
read -sp "SMPL password: " smpl_pass && echo
smpl_user=$(urle "$smpl_user")
smpl_pass=$(urle "$smpl_pass")

wget --post-data "username=$smpl_user&password=$smpl_pass" \
    'https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.1.0.zip' \
    -O /tmp/smpl.zip --no-check-certificate --continue
mkdir -p /tmp/smpl_unzip
unzip -o /tmp/smpl.zip -d /tmp/smpl_unzip
mkdir -p "$PHMR_ROOT/data/body_models/smpl"
mv /tmp/smpl_unzip/SMPL_python_v.1.1.0/smpl/models/basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl \
    "$PHMR_ROOT/data/body_models/smpl/SMPL_NEUTRAL.pkl"
mv /tmp/smpl_unzip/SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl \
    "$PHMR_ROOT/data/body_models/smpl/SMPL_FEMALE.pkl"
mv /tmp/smpl_unzip/SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl \
    "$PHMR_ROOT/data/body_models/smpl/SMPL_MALE.pkl"
rm -rf /tmp/smpl_unzip /tmp/smpl.zip

# Download supplementary SMPL-X data from Hugging Face
wget -O /tmp/supplementary_data.zip \
  https://huggingface.co/daviddobas/video2robot-hackathon-data/resolve/main/supplementary_data.zip

# Unzip directly into PHMR data directory
unzip -o /tmp/supplementary_data.zip -d "$PHMR_ROOT/data"

# GMR symlink
mkdir -p "$GMR_ROOT/assets/body_models"
if [ ! -L "$GMR_ROOT/assets/body_models/smplx" ] && \
   [ ! -d "$GMR_ROOT/assets/body_models/smplx" ]; then
    ln -s "$PHMR_ROOT/data/body_models/smplx" "$GMR_ROOT/assets/body_models/smplx"
    echo "GMR symlink created."
fi

echo ""
echo "Body models downloaded successfully."
echo "You're ready to run the pipeline!"
