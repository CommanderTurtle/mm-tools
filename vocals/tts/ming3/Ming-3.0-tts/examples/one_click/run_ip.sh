#!/usr/bin/env bash
# Ming-3.0-tts-tiny: a built-in voice, no reference audio
# Every variable below can be overridden from the environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-inclusionAI/Ming-3.0-tts-tiny}"
OUT="${OUT:-${ROOT}/outputs/ip.wav}"
IP="${IP:-UK_Charlotte}"
TEXT="${TEXT:-The garden will remain open until sunset, provided the weather stays clear.}"
mkdir -p "$(dirname -- "${OUT}")"
cd "${ROOT}"
exec "${PYTHON:-python}" infer_tiny.py --model_path "${MODEL}" --task ip \
  --ip "${IP}" \
  --text "${TEXT}" \
  --output_path "${OUT}"
