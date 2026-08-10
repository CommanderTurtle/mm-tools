#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

for command_name in uv ffmpeg; do
  command -v "$command_name" >/dev/null 2>&1 || {
    printf 'Missing required command: %s\n' "$command_name" >&2
    exit 1
  }
done

if [[ ! -f .env ]]; then
  cp .env.local.example .env
  printf 'Created %s/.env from the local template.\n' "$ROOT"
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

: "${PYTHON_VERSION:=3.13.12}"
: "${UV_TORCH_BACKEND:=cu130}"
: "${LONGCAT_MODEL_PATH:?Set LONGCAT_MODEL_PATH in .env}"
: "${LONGCAT_TOKENIZER_PATH:?Set LONGCAT_TOKENIZER_PATH in .env}"

[[ -s "$LONGCAT_MODEL_PATH/config.json" && -s "$LONGCAT_MODEL_PATH/model.safetensors" ]] || {
  printf 'LongCat model directory is incomplete: %s\n' "$LONGCAT_MODEL_PATH" >&2
  exit 1
}
for tokenizer_file in tokenizer_config.json tokenizer.json spiece.model; do
  [[ -s "$LONGCAT_TOKENIZER_PATH/$tokenizer_file" ]] || {
    printf 'Tokenizer file is missing: %s/%s\n' "$LONGCAT_TOKENIZER_PATH" "$tokenizer_file" >&2
    printf 'Run ./download-tokenizer.sh or edit LONGCAT_TOKENIZER_PATH.\n' >&2
    exit 1
  }
done

uv venv --python "$PYTHON_VERSION" --seed
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --torch-backend "$UV_TORCH_BACKEND" -r requirements-local.txt

printf '\nLongCat is ready. Browser UI:\n  ./startwithuv.sh\n'
printf 'CLI:\n  uv run --active --no-sync python -m local_tts.cli "Text to speak"\n'
