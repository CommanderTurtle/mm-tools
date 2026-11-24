#!/usr/bin/env bash
# Install the Sophon-Realtime E2E build into two virtual environments next to this repository:
#   .venv-thinker  vLLM 0.29 + duplexjev-vllm   (the thinker)
#   .venv-engine   speaking engine + gateway
# and download the model to MODEL_DIR (default ./Sophon-Realtime-32B-E2E).
# Uses uv when it is installed, otherwise python -m venv + pip. A PyPI mirror can be set with PIP_INDEX_URL.
# usage: [PYTHON=python3.10] [PIP_INDEX_URL=...] [SKIP_DOWNLOAD=1] bash scripts/install.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); cd "$HERE"
PY=${PYTHON:-python3}
MODEL_DIR=${MODEL_DIR:-$HERE/Sophon-Realtime-32B-E2E}
"$PY" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"'
[ -n "${PIP_INDEX_URL:-}" ] && export UV_INDEX_URL="$PIP_INDEX_URL"

if command -v uv > /dev/null 2>&1; then
  mkvenv() { [ -x "$1/bin/python" ] || uv venv -q -p "$PY" "$1"; }
  pipi() { local v=$1; shift; uv pip install -q --python "$v/bin/python" "$@"; }
else
  mkvenv() { [ -x "$1/bin/python" ] || "$PY" -m venv "$1"; "$1/bin/python" -m pip install -q -U pip wheel; }
  pipi() { local v=$1; shift; "$v/bin/python" -m pip install -q "$@"; }
fi

echo "[1/3] thinker environment (.venv-thinker)"
mkvenv .venv-thinker
pipi .venv-thinker -c scripts/constraints-thinker.txt "vllm[audio]==0.29.0" "duplexjev-vllm==0.3.0"

echo "[2/3] engine environment (.venv-engine)"
mkvenv .venv-engine
pipi .venv-engine -c scripts/constraints-engine.txt -r requirements-engine.txt modelscope

echo "[3/3] model"
if [ "${SKIP_DOWNLOAD:-0}" != "1" ]; then
  .venv-engine/bin/modelscope download --model AdventistsAI/Sophon-Realtime-32B-E2E --local_dir "$MODEL_DIR" --exclude 'thinker/*'   # thinker/ is only for the vLLM-Omni deployment (e2e/)
fi
echo
echo "Done. Start with:"
echo "  SOPHON_MODEL_DIR=$MODEL_DIR THINKER_GPU=0 TALKER_GPU=1 bash scripts/start.sh"
