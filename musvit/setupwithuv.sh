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

if ! command -v pdftoppm >/dev/null 2>&1; then
  command -v apt-get >/dev/null 2>&1 || {
    printf 'MuSViT requires pdftoppm. Install the Poppler utilities for this Linux distribution.\n' >&2
    exit 1
  }
  apt_command=(apt-get)
  if (( EUID != 0 )); then
    command -v sudo >/dev/null 2>&1 || {
      printf 'MuSViT requires poppler-utils; rerun setup as root or install it manually.\n' >&2
      exit 1
    }
    apt_command=(sudo apt-get)
  fi
  "${apt_command[@]}" update
  "${apt_command[@]}" install -y poppler-utils
fi

uv pip install -r requirements-local.txt

printf '\nMuSViT is ready in %s/.venv (Python 3.12.10, torch backend: %s).\n' \
  "$ROOT" "$torch_backend"
printf 'Start it with ./startwithuv.sh\n'
