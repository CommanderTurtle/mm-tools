#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -f .env ]] || { printf 'Missing .env. Run ./uvsetup.sh first.\n' >&2; exit 1; }
[[ -x .venv/bin/python ]] || { printf 'Missing .venv. Run ./uvsetup.sh first.\n' >&2; exit 1; }

set -a
# shellcheck disable=SC1091
source .env
set +a
# shellcheck disable=SC1091
source .venv/bin/activate

exec uv run --active --no-sync python -m uvicorn local_tts.server:app \
  --host "${LONGCAT_HOST:-0.0.0.0}" \
  --port "${LONGCAT_PORT:-8230}"
