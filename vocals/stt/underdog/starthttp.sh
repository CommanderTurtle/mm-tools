#!/usr/bin/env bash
set -euo pipefail
exec bash "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")/serve.sh" http
