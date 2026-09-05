#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

HOST="${CW2_HOST:-0.0.0.0}"
PORT="${CW2_PORT:-8172}"
SCHEME=http
CURL_TLS=()
UVICORN_TLS=()
if [[ -n "${CW2_TLS_CERTFILE:-}" || -n "${CW2_TLS_KEYFILE:-}" ]]; then
  [[ -f "${CW2_TLS_CERTFILE:-}" && -f "${CW2_TLS_KEYFILE:-}" ]] || {
    printf 'CW2_TLS_CERTFILE and CW2_TLS_KEYFILE must both name readable files.\n' >&2
    exit 1
  }
  SCHEME=https
  CURL_TLS=(--insecure)
  UVICORN_TLS=(--ssl-certfile "$CW2_TLS_CERTFILE" --ssl-keyfile "$CW2_TLS_KEYFILE")
fi
LOCAL_URL="${SCHEME}://127.0.0.1:${PORT}"

service_ready() {
  curl "${CURL_TLS[@]}" --fail --silent --show-error --max-time 2 \
    "${LOCAL_URL}/api/health" >/dev/null 2>&1
}

port_is_listening() {
  command -v ss >/dev/null 2>&1 &&
    ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .
}

if service_ready; then
  printf 'CrisperWhisper HTTP service is already running: %s\n' "$LOCAL_URL"
  exit 0
fi
if port_is_listening; then
  printf 'Port %s is already occupied by a different or not-yet-ready service.\n' "$PORT" >&2
  exit 1
fi

[[ -x .venv/bin/python ]] || { printf 'Missing .venv. Run ./uvsetup.sh first.\n' >&2; exit 1; }
command -v flock >/dev/null 2>&1 || { printf 'Missing flock (util-linux).\n' >&2; exit 1; }
LOCK_FILE="${XDG_RUNTIME_DIR:-/tmp}/crisperwhisper-local-${UID}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'CrisperWhisper HTTP service is already starting in another terminal: %s\n' "$LOCAL_URL"
  exit 0
fi

if service_ready; then
  printf 'CrisperWhisper HTTP service is already running: %s\n' "$LOCAL_URL"
  exit 0
fi
if port_is_listening; then
  printf 'Port %s became occupied before CrisperWhisper could start.\n' "$PORT" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export CW2_APP_ROLE=http
exec uv run --active --no-sync python -m uvicorn local_app.server:app \
  --host "$HOST" \
  --port "$PORT" \
  "${UVICORN_TLS[@]}"
