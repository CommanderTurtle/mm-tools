#!/bin/bash
# Sophon-Realtime-32B-E2E on vLLM-Omni — one-time install: Python 3.12 venv + vLLM 0.30 + vLLM-Omni 0.30 + the Sophon plugin.
# usage: bash install.sh          env: SIZE=4B (the 4B model, one 32 GB GPU)  PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple (mirror)  SKIP_DOWNLOAD=1 (weights already in ./model)
set -e
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
IDX=${PIP_INDEX:+--index-url $PIP_INDEX}
[ -n "$PIP_INDEX" ] && export UV_DEFAULT_INDEX=$PIP_INDEX UV_INDEX_URL=$PIP_INDEX PIP_INDEX_URL=$PIP_INDEX
export PATH=$HOME/.local/bin:$HERE/.uvboot/bin:$PATH
if ! command -v uv >/dev/null; then   # get uv: pip --user, else a throwaway venv, else the official installer
  python3 -m pip install --user -q uv $IDX 2>/dev/null || { python3 -m venv .uvboot 2>/dev/null && .uvboot/bin/pip install -q uv $IDX; } || curl -LsSf https://astral.sh/uv/install.sh | sh
fi
command -v uv >/dev/null || { echo "could not install uv"; exit 1; }
uv python install 3.12
[ -d .venv ] || uv venv --python 3.12 --seed .venv
source .venv/bin/activate
uv pip install $IDX vllm==0.30.0 ${TORCH_BACKEND:+--torch-backend=$TORCH_BACKEND}
uv pip install $IDX vllm-omni==0.30.0 duplexjev-vllm==0.3.0
uv pip install $IDX -e ./plugin
# the voice stage needs torch 2.14.1 (with an older torch its compiled graphs take about 50 GB more GPU memory); install it last so nothing downgrades it
uv pip install $IDX torch==2.14.1 torchaudio==2.11.0 torchvision==0.29.2 ${TORCH_BACKEND:+--torch-backend=$TORCH_BACKEND}
# model weights (~41 GB) -> ./model ; skip with SKIP_DOWNLOAD=1 or point start.sh at MODEL_DIR=...
# (base/ and adapters/ in the same repository belong to the own-engine deployment in ../scripts and are not needed here)
if [ "${SKIP_DOWNLOAD:-0}" != 1 ] && [ ! -f model/talker220/config.json ]; then
  uv pip install $IDX modelscope
  modelscope download --model ${MODEL_REPO:-AdventistsAI/Sophon-Realtime-${SIZE:-32B}-E2E} --local_dir model --exclude 'base/*' 'adapters/*'
fi
# voice cloning (1.3.0): two small models from the Tag release turn a reference clip into the decoder's prompt (~1 GB); skip with SKIP_CLONE=1
if [ "${SKIP_DOWNLOAD:-0}" != 1 ] && [ "${SKIP_CLONE:-0}" != 1 ] && [ ! -f model/clone_frontend/talker/campplus.onnx ]; then
  uv pip install $IDX modelscope
  modelscope download --model AdventistsAI/Sophon-Realtime-32B-Tag --local_dir model/clone_frontend --include 'talker/speech_tokenizer_v3.onnx' 'talker/campplus.onnx'
fi
python -c "import torch, vllm, vllm_omni, sophon_realtime; assert torch.__version__.startswith('2.14.1'), torch.__version__; print('OK torch', torch.__version__, '| vllm', vllm.__version__, '| vllm-omni', vllm_omni.__version__, '| plugin', sophon_realtime.__version__)"
echo "INSTALL_DONE — now: bash start.sh"
