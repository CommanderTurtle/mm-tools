#!/usr/bin/env bash
# Ming-3.0-tts-tiny: German, Spanish, French, Korean or Japanese in a cloned voice
# Every variable below can be overridden from the environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-inclusionAI/Ming-3.0-tts-tiny}"
OUT="${OUT:-${ROOT}/outputs/multilingual.wav}"
PROMPT_WAV="${PROMPT_WAV:-data/voice_presets/voice_audio/lingguang.wav}"
PROMPT_TEXT="${PROMPT_TEXT:-诶，你今天有空吗？我们能不能聊聊这件事啊？}"
TARGET_LANGUAGE="${TARGET_LANGUAGE:-Japanese}"
TEXT="${TEXT:-週末、駅前の小さな喫茶店で、季節限定の抹茶パフェを食べました}"
mkdir -p "$(dirname -- "${OUT}")"
cd "${ROOT}"
exec "${PYTHON:-python}" infer_tiny.py --model_path "${MODEL}" --task multilingual \
  --target_language "${TARGET_LANGUAGE}" \
  --prompt_wav "${PROMPT_WAV}" --prompt_text "${PROMPT_TEXT}" \
  --text "${TEXT}" \
  --output_path "${OUT}"
