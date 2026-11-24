#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

[[ -x .venv/bin/python ]] || { printf 'Run bash ./setupwithuv.sh first.\n' >&2; exit 1; }
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
# shellcheck disable=SC1091
source .venv/bin/activate

export HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false

exec python -m uvicorn object_remover.server:app \
  --host "${IDEOGRAM_HOST:-0.0.0.0}" \
  --port "${IDEOGRAM_PORT:-8174}"
