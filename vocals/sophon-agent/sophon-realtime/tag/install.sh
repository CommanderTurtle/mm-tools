#!/bin/bash
# Sophon-Realtime-32B-Tag (vLLM-Omni) — one-time install: Python 3.12 venv + vLLM 0.30 + vLLM-Omni 0.30 + the Sophon plugin.
# usage: bash install.sh          env: SIZE=4B (the 4B model, one 32 GB GPU) or SIZE=4B-W4 (4-bit thinker, one 24 GB GPU)  PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple (mirror)  NO_TRT=1 (skip TensorRT)  SKIP_DOWNLOAD=1 (weights already in ./model)
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
uv pip install $IDX vllm==0.30.0 ${TORCH_BACKEND:+--torch-backend=$TORCH_BACKEND}   # default: torch from the same index as vLLM; TORCH_BACKEND=auto = pytorch.org CUDA build
uv pip install $IDX vllm-omni==0.30.0 duplexjev-vllm==0.3.0
if [ "${NO_TRT:-0}" != 1 ]; then   # TensorRT for the flow decoder (about 2x faster speech); optional
  uv pip install $IDX --no-deps tensorrt-cu13==10.13.3.9 tensorrt-cu13-bindings==10.13.3.9 tensorrt-cu13-libs==10.13.3.9 || echo "WARNING: TensorRT could not be installed; the voice will run in plain PyTorch, about 3x slower under load"
fi
uv pip install $IDX -e ./plugin
# model weights (~42 GB) -> ./model ; skip with SKIP_DOWNLOAD=1 or point start.sh at MODEL_DIR=...
if [ "${SKIP_DOWNLOAD:-0}" != 1 ] && [ ! -f model/config.json ]; then
  uv pip install $IDX modelscope
  case "${SIZE:-32B}" in 4B-W4|4b-w4) REPO=Sophon-Realtime-4B-Tag-W4;; *) REPO=Sophon-Realtime-${SIZE:-32B}-Tag;; esac
  modelscope download --model ${MODEL_REPO:-AdventistsAI/$REPO} --local_dir model
fi
python -c "import vllm, vllm_omni, sophon_realtime; print('OK vllm', vllm.__version__, '| vllm-omni', vllm_omni.__version__, '| plugin', sophon_realtime.__version__)"
echo "INSTALL_DONE — now: bash start.sh"
