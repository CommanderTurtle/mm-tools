#!/usr/bin/env bash
set -euo pipefail
# No analytics leaves this studio: hub and gradio telemetry stay off.
export HF_HUB_DISABLE_TELEMETRY=1
export GRADIO_ANALYTICS_ENABLED=False
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
command -v uv >/dev/null || { printf 'Install uv: https://docs.astral.sh/uv/\n' >&2; exit 1; }
command -v bun >/dev/null || { printf 'Install bun: https://bun.sh/\n' >&2; exit 1; }
uv run --no-project --python 3.12 "$ROOT/setup.py"
UV_PROJECT_ENVIRONMENT="$ROOT/.venv" uv sync --locked --no-dev --project "$ROOT"
uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements-http.txt"
bun install --cwd "$ROOT" --frozen-lockfile --ignore-scripts
printf 'Ready. Run %s/startwithuv.sh\n' "$ROOT"
