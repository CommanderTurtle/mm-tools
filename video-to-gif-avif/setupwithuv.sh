#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || {
  printf 'uv is required and was not found on PATH.\n' >&2
  exit 1
}

compute="${1:-${COMPUTE_TARGET:-gpu}}"

case "${compute,,}" in
  gpu)
    torch_backend="${UV_TORCH_BACKEND:-auto}"
    ;;
  cpu)
    torch_backend="cpu"
    ;;
  *)
    printf 'Choose either gpu or cpu.\n' >&2
    exit 2
    ;;
esac

export UV_TORCH_BACKEND="$torch_backend"

if [[ ! -f .env ]]; then
  for template in .env.local.example .env.example; do
    if [[ -f "$template" ]]; then
      cp "$template" .env
      printf 'Created %s/.env from %s; review local model paths before first inference.\n' \
        "$ROOT" "$template"
      break
    fi
  done
fi

uv venv --python 3.12.10 --seed --managed-python
# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install -r requirements.txt

printf '\nVideo to GIF/AVIF is ready in %s/.venv (Python 3.12.10, torch backend: %s).\n' \
  "$ROOT" "$torch_backend"
printf 'Start it with ./startwithuv.sh\n'
