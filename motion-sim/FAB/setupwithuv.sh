#!/usr/bin/env bash
set -euo pipefail
STUDIO="$PWD"
source "$STUDIO/../../models/studio-env.sh"
source "$STUDIO/env.sh"
studio_setup 3.12
uv pip install --python "$VIRTUAL_ENV/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$VIRTUAL_ENV/bin/python" -e "$STUDIO/fab[retarget]" "$STUDIO/../nvidia-sim/SOMA-X[smpl]" 'gradio>=5.49,<6' 'pyrender>=0.1.45' imageio-ffmpeg
# FAH declares top-level bench/train/scripts packages. Keep the two command environments separate.
if [[ ! -x "$STUDIO/.hands/bin/python" ]]; then uv venv --python 3.12 --seed "$STUDIO/.hands"; fi
uv pip install --python "$STUDIO/.hands/bin/python" torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$STUDIO/.hands/bin/python" -e "$STUDIO/fah"
"$VIRTUAL_ENV/bin/python" "$STUDIO/../../models/fab_assets.py"
