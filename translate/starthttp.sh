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

PORT="${TRANSLATE_PORT:-8176}"
LOCAL_URL="http://127.0.0.1:${PORT}"
CURL_AUTH=()
if [[ -n "${TRANSLATE_API_KEY:-}" ]]; then
  CURL_AUTH=(-H "Authorization: Bearer ${TRANSLATE_API_KEY}")
fi

service_ready() {
  curl "${CURL_AUTH[@]}" --fail --silent --show-error --max-time 2 \
    "${LOCAL_URL}/health" >/dev/null 2>&1
}

port_is_listening() {
  command -v ss >/dev/null 2>&1 &&
    ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .
}

if service_ready; then
  printf 'Translate is already running and its model state will be reused: %s\n' "$LOCAL_URL"
  exit 0
fi
if port_is_listening; then
  printf 'Port %s is already occupied by a different or not-yet-ready service.\n' "$PORT" >&2
  exit 1
fi

[[ -x .venv/bin/python ]] || { printf 'Missing .venv. Run ./setupwithuv.sh first.\n' >&2; exit 1; }
command -v flock >/dev/null 2>&1 || { printf 'Missing flock (util-linux).\n' >&2; exit 1; }
LOCK_FILE="${XDG_RUNTIME_DIR:-/tmp}/translate-local-${UID}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'Translate is already starting in another terminal: %s\n' "$LOCAL_URL"
  exit 0
fi

if service_ready; then
  printf 'Translate is already running and its model state will be reused: %s\n' "$LOCAL_URL"
  exit 0
fi
if port_is_listening; then
  printf 'Port %s became occupied before Translate could start.\n' "$PORT" >&2
  exit 1
fi

exec .venv/bin/python server.py "$@"
