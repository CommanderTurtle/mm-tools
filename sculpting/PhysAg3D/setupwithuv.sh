#!/usr/bin/env bash
set -euo pipefail
ROOT="$PWD"
TRELLIS="$ROOT/../trellis"
for tool in uv git bun; do
  command -v "$tool" >/dev/null || { printf '%s is required.\n' "$tool" >&2; exit 1; }
done
export HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 OTEL_SDK_DISABLED=true
[[ -x "$ROOT/.venv/bin/python" ]] || uv venv "$ROOT/.venv" --python 3.12 --seed
uv pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements.txt"
"$ROOT/.venv/bin/python" "$ROOT/setup.py"
if [[ -n "${PHYS_TRELLIS_PYTHON:-}" ]]; then
  NATIVE_PYTHON="$PHYS_TRELLIS_PYTHON"
elif [[ -x "$TRELLIS/.venv/bin/python" ]]; then
  NATIVE_PYTHON="$TRELLIS/.venv/bin/python"
else
  if ! command -v nvcc >/dev/null; then
    for cuda_root in /usr/local/cuda /usr/local/cuda-*; do
      if [[ -x "$cuda_root/bin/nvcc" ]]; then
        export CUDA_HOME="$cuda_root" PATH="$cuda_root/bin:$PATH"
        break
      fi
    done
  fi
  for tool in nvcc g++ perl; do
    command -v "$tool" >/dev/null || { printf '%s is required to build the TRELLIS kernels.\n' "$tool" >&2; exit 1; }
  done
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" MAX_JOBS="${MAX_JOBS:-2}"
  export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
  if [[ -d /usr/lib/wsl/lib ]]; then
    export LIBRARY_PATH="/usr/lib/wsl/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
  [[ -x "$ROOT/.venv-native/bin/python" ]] || uv venv "$ROOT/.venv-native" --python 3.12 --seed
  NATIVE_PYTHON="$ROOT/.venv-native/bin/python"
  uv pip install --python "$NATIVE_PYTHON" --index-url https://download.pytorch.org/whl/cu130 \
    'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' 'torchaudio==2.11.0+cu130'
  uv pip install --python "$NATIVE_PYTHON" 'setuptools<81' wheel packaging ninja -r "$TRELLIS/requirements-local.txt"
  uv pip install --python "$NATIVE_PYTHON" 'natten==0.21.6+torch2110cu130' -f https://whl.natten.org
  header="$("$NATIVE_PYTHON" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "include/ATen/core/List_inl.h")')"
  if grep -Fq 'static_cast<typename decltype(impl_->list)::difference_type>(pos)' "$header"; then
    perl -0pi -e 's/static_cast<typename decltype\(impl_->list\)::difference_type>\(pos\)/static_cast<c10::detail::ListImpl::list_type::difference_type>(pos)/g' "$header"
  fi
  for project in native/nvdiffrast native/nvdiffrec native/CuMesh native/FlexGEMM o-voxel; do
    uv pip install --python "$NATIVE_PYTHON" --no-build-isolation --no-deps "$TRELLIS/$project"
  done
  uv pip install --python "$NATIVE_PYTHON" --no-deps "$TRELLIS/native/utils3d"
fi
"$ROOT/.venv/bin/python" "$ROOT/setup.py" --native-python "$NATIVE_PYTHON"
(cd "$ROOT" && bun install --frozen-lockfile --ignore-scripts)
printf '\nPhysAg3D is ready. Configure omp and Blender on PATH, then run ./startwithuv.sh\n'
