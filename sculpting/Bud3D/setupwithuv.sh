#!/usr/bin/env bash
set -euo pipefail
STUDIO="$PWD"
source "$STUDIO/../../models/studio-env.sh"
studio_setup 3.12
uv pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.9.1' 'torchvision==0.24.1'
uv pip install -r budget3d/requirements.txt gradio==6.0.2
