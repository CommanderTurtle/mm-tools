#!/usr/bin/env bash
set -euo pipefail
# No analytics leaves this studio: hub and gradio telemetry stay off.
export HF_HUB_DISABLE_TELEMETRY=1
export GRADIO_ANALYTICS_ENABLED=False
cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
for tool in uv git bun nvcc ffmpeg; do
  command -v "$tool" >/dev/null || { printf 'Install %s before setup.\n' "$tool" >&2; exit 1; }
done
uv venv --python 3.12 --seed --allow-existing .venv
uv run --no-project --python .venv/bin/python setup.py
uv pip install --python .venv/bin/python torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements.txt
MAX_JOBS="${MAX_JOBS:-2}" uv pip install --python .venv/bin/python --no-build-isolation flash-attn==2.8.3
uv venv --python 3.12 --seed --allow-existing .pe-venv
uv pip install --python .pe-venv/bin/python torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .pe-venv/bin/python -r requirements-pe.txt
bun install --frozen-lockfile --ignore-scripts
printf 'Studio: ./startwithuv.sh\n'
