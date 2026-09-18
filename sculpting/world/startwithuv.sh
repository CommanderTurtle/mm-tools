#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCULPT_ROOT="$(cd -- "$ROOT/.." && pwd)"
VENV="$ROOT/.venv"
VENDOR="$SCULPT_ROOT/.runtime/vendor"
[[ -x "$VENV/bin/python" ]] || { printf 'Run bash ./setupwithuv.sh first.\n' >&2; exit 1; }
# shellcheck disable=SC1090
source "$VENV/bin/activate"

for env_file in "$SCULPT_ROOT/.env.local" "$ROOT/.env.local"; do
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
done

export PYTHONPATH="$SCULPT_ROOT/..:$SCULPT_ROOT:$ROOT:$VENDOR/NAF:$VENDOR/MoGe${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
export SPARSE_CONV_BACKEND="${SPARSE_CONV_BACKEND:-flex_gemm}"
export PIXAL_NAF_SOURCE="$VENDOR/NAF"
export PIXAL_NAF_CHECKPOINT="$SCULPT_ROOT/pretrained/deps/valeoai--NAF/naf_release.pth"
export PIXAL_PIPELINE_CONFIG=pipeline.mmtools.json
export MM_STUDIO_HOST="${WORLD_SCULPT_HOST:-${MM_STUDIO_HOST:-127.0.0.1}}"
export MM_STUDIO_PORT="${WORLD_SCULPT_PORT:-${MM_STUDIO_PORT:-8263}}"
unset MM_STUDIO_API_ONLY

printf 'WorldSculpt Studio: http://%s:%s\n' "$MM_STUDIO_HOST" "$MM_STUDIO_PORT"
exec python "$ROOT/local_app/server.py" \
  --project-root "$ROOT" \
  --manifest "$ROOT/local_app/studio.json" \
  --adapter "$ROOT/local_app/adapter.py" \
  --host "$MM_STUDIO_HOST" \
  --port "$MM_STUDIO_PORT"
