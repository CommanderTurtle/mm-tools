#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

HOST="${NOMAS_HTTP_PROXY_HOST:-0.0.0.0}"
PORT="${NOMAS_HTTP_PROXY_PORT:-8081}"
BACKEND="${NOMAS_HTTP_PROXY_BACKEND:-http://127.0.0.1:18084}"
USERNAME="${NOMAS_DEMO_USERNAME:-hy}"
PASSWORD="${NOMAS_DEMO_PASSWORD:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
NODE_BIN="${NODE_BIN:-$(command -v node || true)}"

if [[ -z "${PASSWORD}" ]]; then
  echo "NOMAS_DEMO_PASSWORD is required" >&2
  exit 2
fi
if [[ -z "${NODE_BIN}" || ! -x "${NODE_BIN}" ]]; then
  echo "Node.js executable not found; set NODE_BIN" >&2
  exit 2
fi

export NOMAS_HTTP_PROXY_BACKEND="${BACKEND}"
export NOMAS_HTTP_PROXY_HOST="${HOST}"
export NOMAS_HTTP_PROXY_PORT="${PORT}"
export NOMAS_DEMO_USERNAME="${USERNAME}"
export NOMAS_DEMO_PASSWORD="${PASSWORD}"

echo "Starting password-protected Hy-Nomas HTTP proxy:"
echo "  listen:  http://${HOST}:${PORT}"
echo "  backend: ${BACKEND}"
echo "  user:    ${USERNAME}"

# This proxy retains Basic Auth for the authoring app while deliberately
# allowing read-only /share/* and /public/* links through without credentials.
exec "${NODE_BIN}" scripts/inference/run_nomas_http_proxy.mjs
