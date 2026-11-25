#!/usr/bin/env bash
set -euo pipefail
STUDIO="$PWD"
source "$STUDIO/../../models/studio-env.sh"
export IDOL_ROOT="$STUDIO/idol-sdk" PYTHONPATH="$STUDIO/budget3d"
export BUDGET3D_DINOV3="$STUDIO/../../models/idol/dinov3" BUDGET3D_FLUX2_VAE="$STUDIO/../../models/idol/flux2"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-8294}"
studio_start studio.py
