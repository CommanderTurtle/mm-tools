#!/usr/bin/env bash
set -euo pipefail
STUDIO="$PWD"
source "$STUDIO/../../models/studio-env.sh"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-8295}"
export PYTHONPATH="$STUDIO/infinisplat/InfiniSplat_v3"
studio_start studio.py
