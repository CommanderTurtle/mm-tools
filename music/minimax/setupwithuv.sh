#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

command -v uv >/dev/null 2>&1 || {
  printf 'uv is required and was not found on PATH.\n' >&2
  exit 1
}

compute="${1:-}"
if [[ -z "$compute" ]]; then
  read -r -p 'Compute target [gpu/cpu] (gpu): ' compute
  compute="${compute:-gpu}"
fi

case "${compute,,}" in
  gpu) torch_backend="${UV_TORCH_BACKEND:-auto}" ;;
  cpu) torch_backend="cpu" ;;
  *)
    printf 'Choose either gpu or cpu.\n' >&2
    exit 2
    ;;
esac

[[ -f .env ]] || cp .env.local.example .env

uv venv --python 3.12.10 --seed --managed-python
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --torch-backend "$torch_backend" torch torchvision torchaudio
uv pip install -r requirements-local.txt

printf '\nMiniMax Music 3 is ready in %s/.venv (Python 3.12.10, torch backend: %s).\n' \
  "$ROOT" "$torch_backend"
printf 'Start it with ./startwithuv.sh\n'
