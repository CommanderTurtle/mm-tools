#!/usr/bin/env bash
# Ming-3.0-tts-tiny: natural-language direction without reference audio
# Every variable below can be overridden from the environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-inclusionAI/Ming-3.0-tts-tiny}"
OUT="${OUT:-${ROOT}/outputs/freeform.wav}"
INSTRUCTION="${INSTRUCTION:-Speak in a warm, intimate storytelling voice, with a gentle pace and a hint of nostalgia.}"
TEXT="${TEXT:-The old radio crackled softly as the storm moved across the valley.}"
mkdir -p "$(dirname -- "${OUT}")"
cd "${ROOT}"
exec "${PYTHON:-python}" infer_tiny.py --model_path "${MODEL}" --task freeform \
  --instruction "${INSTRUCTION}" \
  --text "${TEXT}" \
  --output_path "${OUT}"
