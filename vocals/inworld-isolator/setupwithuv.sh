#!/usr/bin/env bash
set -euo pipefail
STUDIO="$PWD"
source "$STUDIO/../../models/studio-env.sh"
studio_setup 3.11
uv pip install --python "$VIRTUAL_ENV/bin/python" "$STUDIO/voice-isolator" 'soxr>=0.3.7' 'fastapi>=0.115,<1' 'uvicorn[standard]>=0.34,<1' python-multipart
