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
  gpu) export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}" ;;
  cpu) export UV_TORCH_BACKEND=cpu ;;
  *)
    printf 'Choose either gpu or cpu.\n' >&2
    exit 2
    ;;
esac

if [[ ! -f .env ]]; then
  cp .env.local.example .env
fi

uv venv --python 3.12.10 --seed --managed-python .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install torch torchvision torchaudio
uv pip install -r requirements-workstation.txt
uv pip install paddlepaddle==3.1.0
uv pip install --no-deps diffusers
uv pip install 'sdnq>=0.1.5'
python -c 'import pathlib, site, sys; pathlib.Path(site.getsitepackages()[0], "redesign-native.pth").write_text(sys.argv[1] + "\n")' "$ROOT/modules"

if [[ -f modules/grounding_dino/setup.py ]]; then
  uv pip install -e modules/grounding_dino --no-build-isolation ||
    printf 'GroundingDINO CUDA extension was skipped; its Python fallback remains available.\n' >&2
fi

printf 'ReDesign is ready. Start it with ./startwithuv.sh\n'
