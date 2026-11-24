#!/usr/bin/env bash
set -euo pipefail
STUDIO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$STUDIO/../../../models/studio-env.sh"
studio_setup 3.12
uv pip install --python "$VIRTUAL_ENV/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VIRTUAL_ENV/bin/python" -r "$STUDIO/rmbg/requirements-rmbg3.txt" fastapi uvicorn python-multipart sentencepiece protobuf
