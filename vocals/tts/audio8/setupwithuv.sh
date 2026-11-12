#!/usr/bin/env bash
set -euo pipefail
# No analytics leaves this studio: hub and gradio telemetry stay off.
export HF_HUB_DISABLE_TELEMETRY=1
export GRADIO_ANALYTICS_ENABLED=False
cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
for tool in uv ffmpeg; do
  command -v "$tool" >/dev/null || { printf 'Install %s before setup.\n' "$tool" >&2; exit 1; }
done
uv venv --python 3.12 --seed --allow-existing .venv
uv run --no-project --python .venv/bin/python setup.py
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match -r requirements.txt
printf 'Studio: ./startwithuv.sh\n'
