#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
for tool in uv git bun; do
  command -v "$tool" >/dev/null || { printf '%s is required.\n' "$tool" >&2; exit 1; }
done
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 OTEL_SDK_DISABLED=true
[[ -x "$ROOT/.venv/bin/python" ]] || uv venv "$ROOT/.venv" --python 3.12 --seed
"$ROOT/.venv/bin/python" "$ROOT/setup.py"
uv pip install --python "$ROOT/.venv/bin/python" --index-url https://download.pytorch.org/whl/cu130 \
  'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130'
uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt" "$ROOT/idol-sdk"
(cd "$ROOT" && bun install --frozen-lockfile --ignore-scripts)
printf '\nIdol is ready. Run ./startwithuv.sh\n'
