#!/usr/bin/env bash
set -euo pipefail
STUDIO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$STUDIO/../../models/studio-env.sh"
studio_setup 3.12
uv pip install --python "$VIRTUAL_ENV/bin/python" vllm==0.30.0 vllm-omni==0.30.0 duplexjev-vllm==0.3.0 "$STUDIO/sophon-realtime/tag/plugin" python-multipart soundfile httpx 'websockets>=15' 'uvicorn[standard]'
if [[ ${NO_TRT:-0} != 1 ]]; then
  uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps tensorrt-cu13==10.13.3.9 tensorrt-cu13-bindings==10.13.3.9 tensorrt-cu13-libs==10.13.3.9 || echo 'TensorRT unavailable; using torch flow.' >&2
fi
