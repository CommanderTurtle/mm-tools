#!/usr/bin/env bash
set -euo pipefail
# No analytics leaves this studio: hub and gradio telemetry stay off.
export HF_HUB_DISABLE_TELEMETRY=1
export GRADIO_ANALYTICS_ENABLED=False
cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")"
mode="${1:-gpu}"
case "$mode" in gpu) backend=cu128 ;; cpu) backend=cpu ;; *) printf 'Usage: %s [gpu|cpu]\n' "$0" >&2; exit 1 ;; esac
for tool in uv ffmpeg flock; do
  command -v "$tool" >/dev/null || { printf 'Install %s before setup.\n' "$tool" >&2; exit 1; }
done
uv venv --python 3.11 --seed --allow-existing .venv
.venv/bin/python setup.py "$mode"
uv pip install --python .venv/bin/python "torch==2.8.0+$backend" --index-url "https://download.pytorch.org/whl/$backend"
uv pip install --python .venv/bin/python -r requirements.txt
printf 'Studio: ./startwithuv.sh\nHTTP: ./starthttp.sh\nSettings: .env\n'
