#!/usr/bin/env bash
set -euo pipefail
[[ -x .venv/bin/python && -x .pe-venv/bin/python ]] || { printf 'Run ./setupwithuv.sh first.\n' >&2; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export TOKENIZERS_PARALLELISM=false
exec .venv/bin/python server.py "$@"
