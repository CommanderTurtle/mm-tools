#!/usr/bin/env bash
# Ming-3.0-tts-tiny: a voice from a description, no reference audio
# Every variable below can be overridden from the environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-inclusionAI/Ming-3.0-tts-tiny}"
OUT="${OUT:-${ROOT}/outputs/voice_design.wav}"
STYLE="${STYLE:-成熟沉稳的男性纪录片旁白，吐字清晰，节奏舒缓，语气理性而有厚度。}"
TEXT="${TEXT:-冰川在漫长岁月中缓缓移动，也悄悄记录着气候变化的痕迹。}"
mkdir -p "$(dirname -- "${OUT}")"
cd "${ROOT}"
exec "${PYTHON:-python}" infer_tiny.py --model_path "${MODEL}" --task voice_design \
  --style "${STYLE}" \
  --text "${TEXT}" \
  --output_path "${OUT}"
