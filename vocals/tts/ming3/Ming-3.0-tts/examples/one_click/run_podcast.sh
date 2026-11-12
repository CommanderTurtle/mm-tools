#!/usr/bin/env bash
# Ming-3.0-tts-tiny: a two-speaker dialogue, one reference per speaker
# Every variable below can be overridden from the environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MODEL:-inclusionAI/Ming-3.0-tts-tiny}"
OUT="${OUT:-${ROOT}/outputs/podcast.wav}"
PROMPT_WAV_1="${PROMPT_WAV_1:-data/voice_presets/voice_audio/lingguang.wav}"
PROMPT_TEXT_1="${PROMPT_TEXT_1:-诶，你今天有空吗？我们能不能聊聊这件事啊？}"
PROMPT_WAV_2="${PROMPT_WAV_2:-data/voice_presets/voice_audio/db30.wav}"
PROMPT_TEXT_2="${PROMPT_TEXT_2:-感谢你的认可。}"
TEXT="${TEXT:-speaker_1: 你最近是不是又去那家面馆了？
speaker_2: 对啊，排了半个小时的队，不过真的值得。
speaker_1: 下次也带上我吧。}"
mkdir -p "$(dirname -- "${OUT}")"
cd "${ROOT}"
exec "${PYTHON:-python}" infer_tiny.py --model_path "${MODEL}" --task podcast \
  --seed "${SEED:-1235}" \
  --prompt_wav "${PROMPT_WAV_1}" --prompt_text "${PROMPT_TEXT_1}" \
  --prompt_wav "${PROMPT_WAV_2}" --prompt_text "${PROMPT_TEXT_2}" \
  --text "$(printf '%b' "${TEXT}")" \
  --output_path "${OUT}"
