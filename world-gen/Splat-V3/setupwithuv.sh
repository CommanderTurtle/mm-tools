#!/usr/bin/env bash
set -euo pipefail
STUDIO="$PWD"
source "$STUDIO/../../models/studio-env.sh"
studio_setup 3.12
uv pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.9.1'
uv pip install -r infinisplat/InfiniSplat_v3/requirements.txt gradio==6.0.2
npm ci --ignore-scripts --no-audit --no-fund
