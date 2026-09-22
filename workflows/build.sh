#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  printf 'Usage: ./build.sh [v2v | animate | ltx | all]\n\n'
  printf 'Builds the optional studio environments on top of the shared Comfy\n'
  printf 'checkout in %s. The exported workflow JSONs do not need this;\n' "$ROOT/ComfyUI"
  printf 'they run in any native ComfyUI install.\n'
}

lanes=()
for arg in "$@"; do
  case "$arg" in
    v2v|animate|ltx) lanes+=("$arg") ;;
    all) lanes=(v2v animate ltx) ;;
    *) usage; exit 2 ;;
  esac
done
if (( ${#lanes[@]} == 0 )); then
  lanes=(v2v animate ltx)
fi

for lane in "${lanes[@]}"; do
  case "$lane" in
    v2v) dir="V2V" ;;
    *) dir="$lane" ;;
  esac
  printf '\n=== Building %s ===\n' "$lane"
  bash "$ROOT/$dir/setupwithuv.sh"
done

printf '\nStudios ready. Start one with its startwithuv.sh; the shared\n'
printf 'Comfy checkout stays usable as-is for native workflow runs.\n'
