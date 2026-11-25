#!/usr/bin/env bash
set -euo pipefail
STUDIO="$PWD"
source "$STUDIO/../../models/studio-env.sh"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-8298}"
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
studio_start studio.py
