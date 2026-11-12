#!/usr/bin/env bash
# Ming-3.0-tts-tiny: Cantonese or Sichuanese on a reference voice
# Every variable below can be overridden from the environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-inclusionAI/Ming-3.0-tts-tiny}"
OUT="${OUT:-${ROOT}/outputs/dialect.wav}"
PROMPT_WAV="${PROMPT_WAV:-data/wavs/yue_prompt.wav}"
DIALECT="${DIALECT:-广粤话}"
TEXT="${TEXT:-今日天气几好，我哋食完饭一齐去海边行下啦。}"
mkdir -p "$(dirname -- "${OUT}")"
cd "${ROOT}"
exec "${PYTHON:-python}" infer_tiny.py --model_path "${MODEL}" --task dialect \
  --prompt_wav "${PROMPT_WAV}" \
  --dialect "${DIALECT}" \
  --text "${TEXT}" \
  --output_path "${OUT}"
