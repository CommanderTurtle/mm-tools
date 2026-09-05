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

UI_PORT="${TRANSLATE_UI_PORT:-8177}"
UI_HOST="${TRANSLATE_UI_HOST:-${TRANSLATE_HOST:-0.0.0.0}}"
LOCAL_URL="http://127.0.0.1:${UI_PORT}"

ui_ready() {
  curl --fail --silent --show-error --max-time 2 "${LOCAL_URL}/api/health" >/dev/null 2>&1
}

port_is_listening() {
  command -v ss >/dev/null 2>&1 &&
    ss -H -ltn "sport = :${UI_PORT}" 2>/dev/null | grep -q .
}

if ui_ready; then
  printf 'Translate browser workbench is already running: %s\n' "$LOCAL_URL"
  exit 0
fi
if port_is_listening; then
  printf 'Translate UI port %s is already occupied by another service.\n' "$UI_PORT" >&2
  exit 1
fi

[[ -x .venv/bin/python ]] || { printf 'Missing .venv. Run ./setupwithuv.sh first.\n' >&2; exit 1; }
command -v flock >/dev/null 2>&1 || { printf 'Missing flock (util-linux).\n' >&2; exit 1; }
LOCK_FILE="${XDG_RUNTIME_DIR:-/tmp}/translate-ui-${UID}.lock"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf 'Translate browser workbench is already starting: %s\n' "$LOCAL_URL"
  exit 0
fi

printf 'Translate browser workbench: %s\n' "$LOCAL_URL"
source .venv/bin/activate
exec uv run --active --no-sync python -m uvicorn local_app.server:app \
  --host "$UI_HOST" \
  --port "$UI_PORT"
