#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# CrisperWhisper's existing FastAPI process serves both the browser workbench
# and the HTTP API. Keep one implementation and expose this explicit backend
# name for Vox, vox-http, and other machine clients.
exec "$ROOT/startwithuv.sh" "$@"
