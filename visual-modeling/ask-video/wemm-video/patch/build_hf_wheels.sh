#!/usr/bin/env bash
set -euo pipefail

patch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:?Pass a Docker profile}"
output=/opt/hf-wheels
mkdir -p "$output"
case "$profile" in
  vllm-0.18.0|sglang-0.5.16) exit 0 ;;
  vllm-0.31.0|sglang-0.5.19) ;;
  *) echo "Unknown Docker profile: $profile" >&2; exit 2 ;;
esac

flash=flash_attn-2.8.3.post2-cp312-cp312-linux_x86_64.whl
conv=causal_conv1d-1.7.0-cp312-cp312-linux_x86_64.whl
provided=/opt/hf-wheel-cache
cache=/var/cache/wemm-hf-wheels

if [[ -e "$provided/$flash" || -e "$provided/$conv" ]]; then
  (cd "$provided" && sha256sum --check "$patch_dir/hf-wheels.sha256")
  cp "$provided/$flash" "$provided/$conv" "$output/"
  exit 0
fi

if [[ ! -f "$cache/$flash" || ! -f "$cache/$conv" ]]; then
  # The newer profiles share the same Torch, CUDA, and HF extension ABI.
  lock="$patch_dir/vllm/0.31.0/requirements.lock.txt"
  python -m pip install --index-url https://pypi.org/simple uv==0.12.0
  python -m uv --no-config pip install \
    --default-index https://pypi.org/simple --index https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match --prerelease allow --constraint "$lock" \
    torch==2.13.0+cu129 einops ninja packaging psutil setuptools wheel

  export MAX_JOBS="${HF_BUILD_JOBS:-1}" NVCC_THREADS=1
  export TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0" FLASH_ATTN_CUDA_ARCHS="80;90"
  export FLASH_ATTENTION_FORCE_BUILD=TRUE CAUSAL_CONV1D_FORCE_BUILD=TRUE
  stage="$(mktemp -d "$cache/.build.XXXXXX")"
  trap 'rm -rf "$stage"' EXIT
  python -m pip wheel --index-url https://pypi.org/simple --no-deps --no-build-isolation \
    --no-binary flash-attn,causal-conv1d --wheel-dir "$stage" \
    flash-attn==2.8.3.post2 causal-conv1d==1.7.0
  mv "$stage/$flash" "$stage/$conv" "$cache/"
fi
cp "$cache/$flash" "$cache/$conv" "$output/"
