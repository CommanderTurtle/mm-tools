#!/usr/bin/env bash
set -euo pipefail

patch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:?Pass a Docker profile}"
engine="${profile%-*}"
version="${profile##*-}"
lock="$patch_dir/$engine/$version/requirements.lock.txt"
if [[ ! -f "$lock" ]]; then
  echo "No dependency lock for $profile" >&2
  exit 2
fi

python -m pip install --index-url https://pypi.org/simple uv==0.12.0
case "$profile" in
  vllm-0.31.0|sglang-0.5.19)
    python -m pip install --no-deps /opt/hf-wheels/*.whl
    ;;
esac
python -m uv --no-config pip sync \
  --default-index https://pypi.org/simple --index https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match \
  "$lock"
python "$patch_dir/apply.py" "$engine" "$version"

case "$profile" in
  vllm-0.31.0|sglang-0.5.19)
    python -c 'import torch, flash_attn_2_cuda, causal_conv1d_cuda, fla'
    ;;
esac
