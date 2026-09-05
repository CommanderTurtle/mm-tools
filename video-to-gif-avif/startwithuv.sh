#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
[[ -x .venv/bin/python ]] || { printf 'Run ./uvsetup.sh first.\n' >&2; exit 1; }
[[ -f .env ]] && { set -a; source .env; set +a; }
source .venv/bin/activate
exec uv run --active python -m uvicorn app:app \
  --host "${ANIMATOR_HOST:-0.0.0.0}" --port "${ANIMATOR_PORT:-8241}"
