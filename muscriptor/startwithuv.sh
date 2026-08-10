#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

[[ -f .env ]] || {
  printf 'Missing .env. Run ./uvsetup.sh first.\n' >&2
  exit 1
}
[[ -x .venv/bin/python ]] || {
  printf 'Missing .venv. Run ./uvsetup.sh first.\n' >&2
  exit 1
}

set -a
# shellcheck disable=SC1091
source .env
set +a
# shellcheck disable=SC1091
source .venv/bin/activate

: "${MUSCRIPTOR_MODEL_PATH:?Set MUSCRIPTOR_MODEL_PATH in .env}"
: "${MUSCRIPTOR_DEVICE:=cuda}"
: "${MUSCRIPTOR_HOST:=0.0.0.0}"
: "${MUSCRIPTOR_PORT:=8222}"

args=(
  muscriptor serve
  --model "$MUSCRIPTOR_MODEL_PATH"
  --device "$MUSCRIPTOR_DEVICE"
  --host "$MUSCRIPTOR_HOST"
  --port "$MUSCRIPTOR_PORT"
)
if [[ -n "${MUSCRIPTOR_DTYPE:-}" ]]; then
  args+=(--dtype "$MUSCRIPTOR_DTYPE")
fi

exec uv run --active --no-sync "${args[@]}"
