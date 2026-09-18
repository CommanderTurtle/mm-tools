#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV="$ROOT/.venv"
NATIVE="$ROOT/native"
[[ -x "$VENV/bin/python" ]] || { printf 'Run bash ./setupwithuv.sh first.\n' >&2; exit 1; }
# shellcheck disable=SC1090
source "$VENV/bin/activate"

missing_native="$(python - <<'PY'
import importlib.util

required = ("cumesh", "flex_gemm", "nvdiffrast", "nvdiffrec_render", "o_voxel")
print(" ".join(name for name in required if importlib.util.find_spec(name) is None))
PY
)"
if [[ -n "$missing_native" ]]; then
  printf 'Sculpting native runtime is incomplete (%s). Run bash ./setupwithuv.sh first.\n' "$missing_native" >&2
  exit 1
fi

if [[ -f "$ROOT/.env.local" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env.local"
  set +a
fi

export PYTHONPATH="$ROOT/..:$ROOT:$ROOT/world:$NATIVE/NAF:$NATIVE/MoGe${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
export ATTN_BACKEND="${ATTN_BACKEND:-sdpa}"
export SPARSE_ATTN_BACKEND="${SPARSE_ATTN_BACKEND:-sdpa}"
export SPARSE_CONV_BACKEND="${SPARSE_CONV_BACKEND:-flex_gemm}"
export MM_STUDIO_HOST="${SCULPT_STUDIO_HOST:-${MM_STUDIO_HOST:-127.0.0.1}}"
export MM_STUDIO_PORT="${SCULPT_STUDIO_PORT:-${MM_STUDIO_PORT:-8262}}"
unset MM_STUDIO_API_ONLY

printf 'Sculpting Studio: http://%s:%s\n' "$MM_STUDIO_HOST" "$MM_STUDIO_PORT"
exec python "$ROOT/local_app/server.py" \
  --project-root "$ROOT" \
  --manifest "$ROOT/local_app/studio.json" \
  --adapter "$ROOT/local_app/adapter.py" \
  --host "$MM_STUDIO_HOST" \
  --port "$MM_STUDIO_PORT"
