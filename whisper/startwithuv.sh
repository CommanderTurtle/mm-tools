#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ ! -x .venv/bin/python ]]; then
  printf 'Missing .venv. Run ./uvsetup.sh first.\n' >&2
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

args=(
  python -m uvicorn local_app.server:app
  --host "${CW2_HOST:-0.0.0.0}"
  --port "${CW2_PORT:-8172}"
)
if [[ -n "${CW2_TLS_CERTFILE:-}" || -n "${CW2_TLS_KEYFILE:-}" ]]; then
  [[ -f "${CW2_TLS_CERTFILE:-}" && -f "${CW2_TLS_KEYFILE:-}" ]] || {
    printf 'CW2_TLS_CERTFILE and CW2_TLS_KEYFILE must both name readable files.\n' >&2
    exit 1
  }
  args+=(--ssl-certfile "$CW2_TLS_CERTFILE" --ssl-keyfile "$CW2_TLS_KEYFILE")
fi

exec uv run --active --no-sync "${args[@]}"
