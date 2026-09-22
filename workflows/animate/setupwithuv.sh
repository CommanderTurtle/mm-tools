#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
V2V_VENV="$ROOT/.venv" "$ROOT/../V2V/setupwithuv.sh"
CUSTOM_NODES="$ROOT/../ComfyUI/custom_nodes"
mkdir -p "$CUSTOM_NODES"
ln -sfn "$ROOT/ComfyUI-MMToolsAnimate" "$CUSTOM_NODES/mmtools_animate"
printf '\nWan Animate Studio is ready. Run ./startwithuv.sh.\n'
