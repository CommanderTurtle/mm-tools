#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# LongCat's existing FastAPI process serves both the browser workbench and the
# HTTP API. This explicit backend name delegates to that one implementation.
exec "$ROOT/startwithuv.sh" "$@"
