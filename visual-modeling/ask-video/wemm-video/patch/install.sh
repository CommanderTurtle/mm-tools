#!/usr/bin/env bash
set -euo pipefail

patch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python}"

case "${1:-}:${2:-}" in
  vllm:0.18.0|vllm:0.31.0|sglang:0.5.16|sglang:0.5.19) ;;
  *)
    echo "Usage: bash patch/install.sh {vllm 0.18.0|vllm 0.31.0|sglang 0.5.16|sglang 0.5.19}" >&2
    exit 2
    ;;
esac
if [[ "$#" != 2 ]]; then
  echo "Pass exactly one engine and version." >&2
  exit 2
fi

profile="$patch_dir/$1/$2"
python_path="$("$python_bin" -c '
import platform
import sys

if sys.prefix == sys.base_prefix:
    sys.exit("Activate a dedicated Python 3.12 virtual environment first.")
if sys.version_info[:2] != (3, 12) or sys.platform != "linux" or platform.machine() != "x86_64":
    sys.exit("These wheel profiles require Python 3.12 on Linux x86_64.")
print(sys.executable)
')"

"$python_path" -m pip install --index-url https://pypi.org/simple 'uv==0.12.0'
"$python_path" -m uv --no-config pip install --python "$python_path" \
  --default-index https://pypi.org/simple --index https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match --prerelease allow \
  --overrides "$profile/overrides.txt" -r "$profile/requirements.txt" -r "$patch_dir/media.txt"
"$python_path" "$patch_dir/apply.py" "$1" "$2"
