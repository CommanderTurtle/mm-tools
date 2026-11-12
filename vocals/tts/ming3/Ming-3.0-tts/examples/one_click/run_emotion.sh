#!/usr/bin/env bash
# Ming-3.0-tts-tiny: emotion on a reference voice
# Every variable below can be overridden from the environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-inclusionAI/Ming-3.0-tts-tiny}"
OUT="${OUT:-${ROOT}/outputs/emotion.wav}"
PROMPT_WAV="${PROMPT_WAV:-data/wavs/emotion_prompt.wav}"
EMOTION="${EMOTION:-高兴}"
TEXT="${TEXT:-所有测试都顺利通过了，我们终于可以把这个好消息告诉大家！}"
mkdir -p "$(dirname -- "${OUT}")"
cd "${ROOT}"
exec "${PYTHON:-python}" infer_tiny.py --model_path "${MODEL}" --task emotion \
  --prompt_wav "${PROMPT_WAV}" \
  --emotion "${EMOTION}" \
  --text "${TEXT}" \
  --output_path "${OUT}"
