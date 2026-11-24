#!/usr/bin/env bash
set -euo pipefail
STUDIO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$STUDIO/../../models/studio-env.sh"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-8299}"
source "$STUDIO/env.sh"
export PYOPENGL_PLATFORM=egl MUJOCO_GL=egl
studio_start studio.py
