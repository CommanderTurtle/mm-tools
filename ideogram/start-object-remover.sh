#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ -x .venv/bin/python ]] || { printf 'Run ./setupwithuv first.\n' >&2; exit 1; }
if [[ -f .env ]]; then set -a; source .env; set +a; fi
# shellcheck disable=SC1091
source .venv/bin/activate
exec uv run --active --no-sync python -m uvicorn object_remover.server:app \
  --host "${OBJECT_REMOVER_HOST:-0.0.0.0}" --port "${OBJECT_REMOVER_PORT:-8174}"
