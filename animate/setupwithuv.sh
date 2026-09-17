#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"$ROOT/../V2V/setupwithuv.sh"
printf '\nWan Animate Studio is ready. Run ./startwithuv.sh.\n'
