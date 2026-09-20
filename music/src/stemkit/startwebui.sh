#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$ROOT/../../.." && pwd)"

# The StemKit runtime has no environment of its own: it runs inside an
# existing music-studio venv that already carries torch.  Honour an explicit
# STEMKIT_VENV first, then prefer the YuE2 environment (its torch pin matches
# the demucs/torchaudio pair installed by the studio setups), then MiniMax.
pick_venv() {
  local candidate
  if [[ -n "${STEMKIT_VENV:-}" ]]; then
    candidate="$STEMKIT_VENV"
  elif [[ -x "$REPO_ROOT/music/yue2/.venv/bin/python" ]]; then
    candidate="$REPO_ROOT/music/yue2/.venv"
  elif [[ -x "$REPO_ROOT/music/minimax/.venv/bin/python" ]]; then
    candidate="$REPO_ROOT/music/minimax/.venv"
  else
    candidate=""
  fi
  if [[ -n "$candidate" && ! -x "$candidate/bin/python" ]]; then
    printf 'StemKit venv not found: %s\nRun a music studio setup first.\n' "$candidate" >&2
    return 1
  fi
  printf '%s' "$candidate"
}

VENV="$(pick_venv)"
if [[ -z "$VENV" ]]; then
  printf 'No usable StemKit environment. Set STEMKIT_VENV or run\n' >&2
  printf 'music/yue2/setupwithuv.sh / music/minimax/setupwithuv.sh first.\n' >&2
  exit 1
fi

"$VENV/bin/python" -c 'import demucs' 2>/dev/null || {
  printf 'demucs is missing from %s; rerun that studio setup.\n' "$VENV" >&2
  exit 1
}

HOST="${STEMKIT_STUDIO_HOST:-127.0.0.1}"
PORT="${STEMKIT_STUDIO_PORT:-8271}"

export HF_HUB_DISABLE_TELEMETRY=1
export DO_NOT_TRACK=1

printf 'StemKit Studio: http://%s:%s (running in %s)\n' "$HOST" "$PORT" "$VENV"
exec "$VENV/bin/python" "$ROOT/local_app/server.py" --host "$HOST" --port "$PORT"
