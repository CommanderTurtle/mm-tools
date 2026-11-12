#!/usr/bin/env bash
set -euo pipefail
# No analytics leaves this studio: hub and gradio telemetry stay off.
export HF_HUB_DISABLE_TELEMETRY=1
export GRADIO_ANALYTICS_ENABLED=False
exec bash "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/serve.sh" ui
