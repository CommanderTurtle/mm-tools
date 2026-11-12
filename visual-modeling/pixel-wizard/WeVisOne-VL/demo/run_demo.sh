#!/usr/bin/env bash
set -euo pipefail

demo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${demo_dir}/.." && pwd)"
cd "${repo_root}"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

model="${WEVISONE_MODEL:-tencent/WeVisOne-VL-4B}"
exec python "${demo_dir}/app.py" --model "${model}" "$@"
