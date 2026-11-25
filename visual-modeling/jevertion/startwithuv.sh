#!/usr/bin/env bash
set -euo pipefail
STUDIO="$PWD"
source "$STUDIO/../../models/studio-env.sh"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-8300}"
export HF_HUB_CACHE="$STUDIO/../../models/jeveration/hub"
export GRADIO_TEMP_DIR="$STUDIO/.state/workflow"
studio_start studio.py
