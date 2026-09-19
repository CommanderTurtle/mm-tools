#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
[[ -x "$VENV/bin/python" ]] || { printf 'Run bash ./setupwithuv.sh first.\n' >&2; exit 1; }
# shellcheck disable=SC1090
source "$VENV/bin/activate"
[[ -x "$ROOT/.venv-sheetsage2/bin/python" ]] || { printf 'Run bash ./setupwithuv.sh to prepare the isolated SheetSage2 lane.\n' >&2; exit 1; }

if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi

export PYTHONPATH="$ROOT/../..:$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export MM_STUDIO_HOST="${YUE2_STUDIO_HOST:-${MM_STUDIO_HOST:-127.0.0.1}}"
export MM_STUDIO_PORT="${YUE2_STUDIO_PORT:-${MM_STUDIO_PORT:-8270}}"
unset MM_STUDIO_API_ONLY

printf 'YuE2 Composition Studio: http://%s:%s\n' "$MM_STUDIO_HOST" "$MM_STUDIO_PORT"
exec python "$ROOT/local_app/server.py" \
  --project-root "$ROOT" \
  --manifest "$ROOT/local_app/studio.json" \
  --adapter "$ROOT/local_app/adapter.py" \
  --host "$MM_STUDIO_HOST" \
  --port "$MM_STUDIO_PORT"
