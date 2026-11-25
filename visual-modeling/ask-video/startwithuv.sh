#!/usr/bin/env bash
set -euo pipefail
STUDIO="$PWD"
source "$STUDIO/../../models/studio-env.sh"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-8302}"
export VLLM_NO_USAGE_STATS=1 VLLM_DO_NOT_TRACK=1
npp="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/npp/lib"
export LD_LIBRARY_PATH="$npp${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
studio_start studio.py
