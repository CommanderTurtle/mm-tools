#!/usr/bin/env bash
set -euo pipefail
STUDIO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$STUDIO/../../models/studio-env.sh"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-8296}"
export PYTHONPATH="$STUDIO/fastframes${PYTHONPATH:+:$PYTHONPATH}"
studio_start studio.py
