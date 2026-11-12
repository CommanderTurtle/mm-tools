#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)"
cd -- "$ROOT"
[[ -x .venv/bin/python && -f .env ]] || { printf 'Run ./setupwithuv.sh first.\n' >&2; exit 1; }
set -a
source .env
set +a
role="${1:-ui}"
case "$role" in
  ui) port="${BARK_UI_PORT:-8293}" ;;
  http) port="${BARK_PORT:-8292}" ;;
  *) printf 'Unknown service role.\n' >&2; exit 1 ;;
esac
export BARK_APP_ROLE="$role" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
tls=()
if [[ -n "${BARK_TLS_CERTFILE:-}" || -n "${BARK_TLS_KEYFILE:-}" ]]; then
  [[ -f "${BARK_TLS_CERTFILE:-}" && -f "${BARK_TLS_KEYFILE:-}" ]] || { printf 'Set both TLS certificate and key paths.\n' >&2; exit 1; }
  tls=(--ssl-certfile "$BARK_TLS_CERTFILE" --ssl-keyfile "$BARK_TLS_KEYFILE")
fi
# One owner per role, including simultaneous launchers. The inherited file
# descriptor releases the lock on graceful or abrupt process termination.
exec 9>"$ROOT/.${role}.lock"
flock -n 9 || { printf 'Underdog %s is already running.\n' "$role" >&2; exit 1; }
exec .venv/bin/python launch.py --host "${BARK_HOST:-127.0.0.1}" --port "$port" "${tls[@]}"
