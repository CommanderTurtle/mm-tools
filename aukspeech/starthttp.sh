#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MM_STUDIO_API_ONLY=1

if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi

# API-specific overrides always win over shared WebUI values, including when
# .env.local defines both profiles.
export MM_STUDIO_API_ONLY=1
export AUK_STUDIO_HOST="${AUK_HTTP_HOST:-${AUK_STUDIO_HOST:-127.0.0.1}}"
export AUK_STUDIO_PORT="${AUK_HTTP_PORT:-${AUK_STUDIO_PORT:-8260}}"

VENV="$ROOT/.venv"
[[ -x "$VENV/bin/python" ]] || { printf 'Run ./setupwithuv.sh first.\n' >&2; exit 1; }
export PYTHONPATH="$ROOT/..:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false

printf 'AuK private API: http://%s:%s (UI disabled)\n' "$AUK_STUDIO_HOST" "$AUK_STUDIO_PORT"
exec "$VENV/bin/python" "$ROOT/../studio/server.py" \
  --project-root "$ROOT" \
  --manifest "$ROOT/local_app/studio.json" \
  --adapter "$ROOT/local_app/adapter.py" \
  --host "$AUK_STUDIO_HOST" \
  --port "$AUK_STUDIO_PORT"
