#!/usr/bin/env bash
set -euo pipefail
# No analytics leaves this studio: hub and gradio telemetry stay off.
export HF_HUB_DISABLE_TELEMETRY=1
export GRADIO_ANALYTICS_ENABLED=False
for tool in uv git bun; do
  command -v "$tool" >/dev/null || { printf 'Install %s before setup.\n' "$tool" >&2; exit 1; }
done
uv venv --python 3.10 --seed --allow-existing .venv
uv run --no-project --python .venv/bin/python setup.py
uv pip install --python .venv/bin/python -r requirements.txt
bun install --frozen-lockfile --ignore-scripts
printf 'Studio: ./startwithuv.sh\n'
