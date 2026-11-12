#!/usr/bin/env bash
# Ming-3.0-tts-tiny: speed, pitch and volume on a reference voice
# Every variable below can be overridden from the environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-inclusionAI/Ming-3.0-tts-tiny}"
OUT="${OUT:-${ROOT}/outputs/basic.wav}"
PROMPT_WAV="${PROMPT_WAV:-data/voice_presets/voice_audio/lingguang.wav}"
SPEED="${SPEED:-慢速}"
PITCH="${PITCH:-中}"
VOLUME="${VOLUME:-中}"
TEXT="${TEXT:-今晚的观测将在云层散开后开始，请耐心等待下一次通知。}"
mkdir -p "$(dirname -- "${OUT}")"
cd "${ROOT}"
exec "${PYTHON:-python}" infer_tiny.py --model_path "${MODEL}" --task basic \
  --prompt_wav "${PROMPT_WAV}" \
  --speed "${SPEED}" --pitch "${PITCH}" --volume "${VOLUME}" \
  --text "${TEXT}" \
  --output_path "${OUT}"
