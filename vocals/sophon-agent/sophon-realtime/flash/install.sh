#!/bin/bash
# Sophon-Realtime-32B-Flash (vLLM-Omni) — one-time install: Python 3.12 venv + vLLM 0.30 + vLLM-Omni 0.30 + the Sophon plugin.
# usage: bash install.sh          env: SIZE=4B (the 4B model, one 24 GB GPU)  PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple (mirror)
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
python -c "import vllm, vllm_omni, sophon_realtime; print('OK vllm', vllm.__version__, '| vllm-omni', vllm_omni.__version__, '| plugin', sophon_realtime.__version__)"
echo "INSTALL_DONE — now: bash start.sh"
