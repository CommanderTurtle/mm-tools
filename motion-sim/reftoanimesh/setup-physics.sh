#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PY="${FLOWHMR_PHC_PYTHON:-$ROOT/.venv-phc/bin/python}"
[[ -x "$PY" ]] || { printf 'Set FLOWHMR_PHC_PYTHON to your Isaac Gym Preview 4 Python 3.8 interpreter.\n' >&2; exit 1; }
command -v uv >/dev/null || { printf 'Install uv first.\n' >&2; exit 1; }
"$PY" "$ROOT/setup.py" --physics-environment check
uv run --no-project --python 3.10 "$ROOT/setup.py" --phc
uv pip install --python "$PY" -r "$ROOT/requirements-physics.txt"
uv pip install --python "$PY" --no-build-isolation --no-deps "$ROOT/flowhmr/local_app/chumpy" "$ROOT/flowhmr/local_app/smplx" "$ROOT/flowhmr/local_app/smplsim"
"$PY" "$ROOT/setup.py" --physics-environment record
printf 'PHC dependencies ready. Keep FLOWHMR_PHC_PYTHON set when starting the studio.\n'
