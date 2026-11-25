#!/usr/bin/env bash
set -euo pipefail
STUDIO="$PWD"
source "$STUDIO/../../models/studio-env.sh"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-8297}"
export PYTHONPATH="$STUDIO/voice-isolator/demo${PYTHONPATH:+:$PYTHONPATH}"
studio_start studio.py
