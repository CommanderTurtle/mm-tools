#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="4.5.1"
ARCHIVE="blender-${VERSION}-linux-x64.tar.xz"
URL="https://download.blender.org/release/Blender4.5/${ARCHIVE}"
TARGET="$ROOT/blender/blender-${VERSION}-linux-x64"

if [[ ! -x "$TARGET/blender" ]]; then
  mkdir -p "$ROOT/blender"
  curl --fail --location --retry 3 "$URL" --output "$ROOT/blender/$ARCHIVE"
  tar -xf "$ROOT/blender/$ARCHIVE" -C "$ROOT/blender"
  rm "$ROOT/blender/$ARCHIVE"
fi

BLENDER_PYTHON="$TARGET/4.5/python/bin/python3.11"
if ! "$BLENDER_PYTHON" -c "import PIL" >/dev/null 2>&1; then
  PIP_CONFIG_FILE="${FIRE3D_PIP_CONFIG_FILE:-/dev/null}" \
    "$BLENDER_PYTHON" -m pip install pillow==12.3.0
fi
"$TARGET/blender" --version | head -n 1
