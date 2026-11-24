#!/usr/bin/env bash
set -euo pipefail
STUDIO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$STUDIO/../../models/studio-env.sh"
for cmd in ffmpeg ffprobe; do command -v "$cmd" >/dev/null || { echo "Install FFmpeg with shared libraries before setup." >&2; exit 1; }; done
studio_setup 3.12
PYTHON="$VIRTUAL_ENV/bin/python" bash "$STUDIO/wemm-video/patch/install.sh" vllm 0.18.0
uv pip install --python "$VIRTUAL_ENV/bin/python" nvidia-npp-cu12==12.4.1.87 python-multipart
