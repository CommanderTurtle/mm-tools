#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

[[ -f .env ]] || { printf 'Missing .env. Run ./uvsetup.sh first.\n' >&2; exit 1; }
set -a
# shellcheck disable=SC1091
source .env
set +a

HOST="${LONGCAT_UI_HOST:-${LONGCAT_HOST:-0.0.0.0}}"
PORT="${LONGCAT_UI_PORT:-8231}"
LOCAL_URL="http://127.0.0.1:${PORT}"

ui_ready() {
  curl --fail --silent --show-error --max-time 2 \
    "${LOCAL_URL}/api/status" >/dev/null 2>&1
}

port_is_listening() {
  command -v ss >/dev/null 2>&1 &&
    ss -H -ltn "sport = :${PORT}" 2>/dev/null | grep -q .
}

if ui_ready; then
  printf 'LongCat browser workbench is already running: %s\n' "$LOCAL_URL"
  exit 0
fi
if port_is_listening; then
  printf 'LongCat UI port %s is already occupied by another service.\n' "$PORT" >&2
  exit 1
fi

[[ -x .venv/bin/python ]] || { printf 'Missing .venv. Run ./uvsetup.sh first.\n' >&2; exit 1; }
command -v flock >/dev/null 2>&1 || { printf 'Missing flock (util-linux).\n' >&2; exit 1; }
LOCK_FILE="${XDG_RUNTIME_DIR:-/tmp}/longcat-ui-${UID}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'LongCat browser workbench is already starting: %s\n' "$LOCAL_URL"
  exit 0
fi
if ui_ready; then
  printf 'LongCat browser workbench is already running: %s\n' "$LOCAL_URL"
  exit 0
fi
if port_is_listening; then
  printf 'LongCat UI port %s became occupied before startup.\n' "$PORT" >&2
  exit 1
fi

# The browser process starts without weights. Its Load button owns an optional
# UI-local model; Check Secondary Load can instead attach to starthttp.sh.
export LONGCAT_APP_ROLE=ui
export LONGCAT_AUTOLOAD=0
# shellcheck disable=SC1091
source .venv/bin/activate
printf 'LongCat browser workbench: %s\n' "$LOCAL_URL"
exec uv run --active --no-sync python -m uvicorn local_tts.server:app \
  --host "$HOST" \
  --port "$PORT"
