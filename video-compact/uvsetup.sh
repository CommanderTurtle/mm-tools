#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ -f .env ]] || cp .env.example .env
set -a; source .env; set +a
[[ -x .venv/bin/python ]] || uv venv --python "${PYTHON_VERSION:-3.13.12}" --seed
source .venv/bin/activate
uv pip install -r requirements.txt
ffmpeg -hide_banner -version >/dev/null
ffprobe -hide_banner -version >/dev/null
printf 'Video Compact is installed in %s/.venv.\n' "$PWD"
