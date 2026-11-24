#!/usr/bin/env bash
set -euo pipefail
STUDIO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$STUDIO/../../models/studio-env.sh"
UV_PROJECT_ENVIRONMENT="$VIRTUAL_ENV" uv sync --frozen --python 3.10 --project "$STUDIO/fastframes" --extra lingbot --extra demo
uv pip install --python "$VIRTUAL_ENV/bin/python" imageio-ffmpeg==0.6.0
