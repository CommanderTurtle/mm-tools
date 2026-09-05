#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  printf 'Missing .venv. Run ./setupwithuv first.\n' >&2
  exit 1
fi
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
# shellcheck disable=SC1091
source .venv/bin/activate

exec python -m uvicorn local_app.server:app \
  --host "${REDESIGN_HOST:-0.0.0.0}" \
  --port "${REDESIGN_PORT:-8173}"
