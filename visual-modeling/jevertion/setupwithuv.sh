#!/usr/bin/env bash
set -euo pipefail
STUDIO="$PWD"
source "$STUDIO/../../models/studio-env.sh"
studio_setup 3.11
uv pip install --python "$VIRTUAL_ENV/bin/python" torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VIRTUAL_ENV/bin/python" -r "$STUDIO/../../models/jeveration/space/requirements.txt" gradio==6.30.0
hub="$STUDIO/../../models/jeveration/hub"
pin_ref() {
  local dir="$hub/models--$1"
  [[ -d "$dir/snapshots/$2" ]] || { echo "Select jeveration in the model downloader first." >&2; return 1; }
  mkdir -p "$dir/refs"; printf '%s' "$2" > "$dir/refs/main"
}
pin_ref lllyasviel--Annotators 982e7edaec38759d914a963c48c4726685de7d96
pin_ref depth-anything--Depth-Anything-V2-Small-hf 5426e4f0f36572d16453bbda7a8389317b1bef99
