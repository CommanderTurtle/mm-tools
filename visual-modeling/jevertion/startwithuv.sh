#!/usr/bin/env bash
set -euo pipefail
STUDIO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$STUDIO/../../models/studio-env.sh"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-8300}"
export HF_HUB_CACHE="$STUDIO/../../models/jeveration/hub"
export GRADIO_TEMP_DIR="$STUDIO/.state/workflow"
studio_start studio.py
