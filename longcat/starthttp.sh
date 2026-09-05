#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

[[ -f .env ]] || { printf 'Missing .env. Run ./uvsetup.sh first.\n' >&2; exit 1; }

set -a
# shellcheck disable=SC1091
source .env
set +a

HOST="${LONGCAT_HOST:-0.0.0.0}"
PORT="${LONGCAT_PORT:-8230}"
LOCAL_URL="http://127.0.0.1:${PORT}"

service_ready() {
  curl --fail --silent --show-error --max-time 2 \
    "${LOCAL_URL}/api/status" >/dev/null 2>&1
}

port_is_listening() {
  command -v ss >/dev/null 2>&1 &&
    ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .
}

if service_ready; then
  printf 'LongCat HTTP service is already running: %s\n' "$LOCAL_URL"
  exit 0
fi

if port_is_listening; then
  printf 'Port %s is already occupied by a different or not-yet-ready service.\n' "$PORT" >&2
  exit 1
fi

[[ -x .venv/bin/python ]] || { printf 'Missing .venv. Run ./uvsetup.sh first.\n' >&2; exit 1; }
command -v flock >/dev/null 2>&1 || { printf 'Missing flock (util-linux).\n' >&2; exit 1; }
LOCK_FILE="${XDG_RUNTIME_DIR:-/tmp}/longcat-local-${UID}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'LongCat HTTP service is already starting in another terminal: %s\n' "$LOCAL_URL"
  exit 0
fi

# Recheck after taking the lifecycle lock so simultaneous launchers cannot both
# enter FastAPI startup and load a second checkpoint.
if service_ready; then
  printf 'LongCat HTTP service is already running: %s\n' "$LOCAL_URL"
  exit 0
fi
if port_is_listening; then
  printf 'Port %s became occupied before LongCat could start.\n' "$PORT" >&2
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export LONGCAT_APP_ROLE=http
exec uv run --active --no-sync python -m uvicorn local_tts.server:app \
  --host "$HOST" \
  --port "$PORT"
