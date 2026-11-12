#!/usr/bin/env bash
set -euo pipefail
# No analytics leaves this studio: hub and gradio telemetry stay off.
export HF_HUB_DISABLE_TELEMETRY=1
export GRADIO_ANALYTICS_ENABLED=False
cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
for tool in uv git ffmpeg ffprobe nvcc; do
  command -v "$tool" >/dev/null || { printf 'Install %s before setup.\n' "$tool" >&2; exit 1; }
done
uv venv --python 3.13 --seed --allow-existing .venv
uv run --no-project --python .venv/bin/python setup.py
UV_PROJECT_ENVIRONMENT="$PWD/.venv" uv sync --project Speech --python 3.13 --extra asr --extra vllm --locked --no-default-groups --no-install-project
uv pip install --python .venv/bin/python --no-deps -e Speech
uv pip install --python .venv/bin/python --no-deps av==19.0.2 opencv-python-headless==5.0.0.93
uv pip install --python .venv/bin/python -r requirements.txt
uv venv --python 3.11 --seed --allow-existing .stem-venv
uv pip install --python .stem-venv/bin/python --index-strategy unsafe-best-match -r requirements-stems.txt
printf 'Studio: ./startwithuv.sh\n'
