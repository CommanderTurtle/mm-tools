#!/usr/bin/env bash
# Real-time FAH on a Mac (Apple silicon): sets up whatever is missing, then runs fah-live on camera 0 with the real-time
# preset (FAH-S, Core ML pose network on the GPU + detector on the Neural Engine, online video mode with the
# crossed-hands identity rule, fingertip step on, no mesh fit; keys: q quit, k keypoints, m masks / parts, f mesh fit,
# v video mode, s snapshot).
#
#   scripts/mac_live.sh --weights <dir> --mano <mano_v1_2.zip>   # first run: setup (~3 min), then the camera
#   scripts/mac_live.sh                                          # later runs: straight to the camera
#   scripts/mac_live.sh --video clip.mp4 --no_window --frames 300   # any other arguments go to fah-live (fah-live --help)
#
# First run, each step skipped once done:
#   1. Python env in .venv (uv, installed to ~/.local/bin if missing; Python 3.11, PyTorch, coremltools, ...)
#   2. FAH weights: --weights DIR / $FAH_WEIGHTS, default $FAH_HOME/weights (FAH_HOME = ~/.cache/fah)
#   3. two SAM 3D Body files (gated: accept the SAM License on Hugging Face, then `.venv/bin/hf auth login`)
#   4. MANO (register at mano.is.tue.mpg.de, download the models, pass --mano <zip or folder>) -> $FAH_HOME/mano
#   5. Core ML exports -> <weights>/coreml
# Steps 2-4 need you: the script lists what is missing and stops. pip instead of uv: FAH_PIP=1 (PYTHON=python3.11).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FAH_HOME="${FAH_HOME:-$HOME/.cache/fah}"
WEIGHTS="${FAH_WEIGHTS:-$FAH_HOME/weights}"
SAM3D="${FAH_SAM3D:-$FAH_HOME/sam-3d-body-dinov3}"
MANO="${MANO_DIR:-$FAH_HOME/mano}"
HF_REPO="${FAH_HF_REPO:-}"  # Hugging Face repository of the FAH weights (set once they are hosted)
MANO_SRC=""
SETUP_ONLY=0
ARGS=()
while (($#)); do
  case "$1" in
    --weights) WEIGHTS="$2"; shift 2 ;;
    --mano) MANO_SRC="$2"; shift 2 ;;
    --setup_only) SETUP_ONLY=1; shift ;;
    -h|--help) sed -n '2,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
die() { printf '\n%s\n' "$@" >&2; exit 1; }

# platform
[[ "$(uname -s)" == Darwin ]] || die "scripts/mac_live.sh is for macOS on Apple silicon (other systems: README, fah-live)."
if [[ "$(uname -m)" != arm64 ]]; then
  [[ "$(sysctl -n sysctl.proc_translated 2>/dev/null || true)" == 1 ]] && die "This terminal runs under Rosetta (x86_64): open a native arm64 terminal and run again."
  die "FAH's real-time path needs Apple silicon (M1 or later)."
fi
COREML=1
if (( $(sw_vers -productVersion | cut -d. -f1) < 15 )); then  # multifunction Core ML models: macOS 15+
  COREML=0
  echo "macOS $(sw_vers -productVersion): the Core ML exports need macOS 15 or later, so PyTorch (MPS, slower) runs instead."
fi

# 1. Python env (reinstalled when pyproject.toml / requirements.txt change)
PY="$REPO/.venv/bin/python"
KEY="$(cat "$REPO/pyproject.toml" "$REPO/requirements.txt" | shasum | cut -d' ' -f1)"
if [[ ! -x "$PY" || "$(cat "$REPO/.venv/.mac_live" 2>/dev/null || true)" != "$KEY" ]]; then
  echo "== Python environment: $REPO/.venv"
  if [[ -n "${FAH_PIP:-}" ]]; then  # pip instead of uv: FAH_PIP=1, PYTHON = a Python 3.10+ (default python3)
    "${PYTHON:-python3}" -c 'import sys; sys.exit(sys.version_info < (3, 10))' ||
      die "FAH_PIP: $("${PYTHON:-python3}" -V 2>&1) is too old; pass a Python 3.10+, e.g. PYTHON=python3.11."
    [[ -x "$PY" ]] || "${PYTHON:-python3}" -m venv "$REPO/.venv"
    "$PY" -m pip install -e "$REPO[coreml]"
  else
    if ! command -v uv >/dev/null 2>&1; then
      if [[ ! -x "$HOME/.local/bin/uv" ]]; then
        echo "== installing uv (https://docs.astral.sh/uv)"
        curl -LsSf https://astral.sh/uv/install.sh | sh
      fi
      export PATH="$HOME/.local/bin:$PATH"
    fi
    [[ -x "$PY" ]] || uv venv --python 3.11 "$REPO/.venv"
    uv pip install --python "$PY" -e "$REPO[coreml]"
  fi
  echo "$KEY" > "$REPO/.venv/.mac_live"
fi

# 2-4. files that need you: collect what is missing, then stop once with all of it
MISSING=()
WFILES=(fah-s/model.pt fah-s/config.json shared/detector.pt shared/video_refiner.pt shared/mesh_fitter.pt)
weights_ok() { for f in "${WFILES[@]}"; do [[ -f "$WEIGHTS/$f" ]] || return 1; done; }
if ! weights_ok && [[ -n "$HF_REPO" ]]; then
  echo "== FAH-S weights from $HF_REPO -> $WEIGHTS"
  "$PY" -c "import sys; from huggingface_hub import snapshot_download as d; d(sys.argv[1], local_dir=sys.argv[2], allow_patterns=sys.argv[3:])" \
    "$HF_REPO" "$WEIGHTS" "${WFILES[@]}"
fi
weights_ok || MISSING+=("FAH weights: not found in $WEIGHTS (needs ${WFILES[*]}).
   Download them from Hugging Face (logged in with hf auth login):
       $REPO/.venv/bin/hf download noahcao/FAH --exclude \"qc/*\" --local-dir $WEIGHTS
   or pass --weights <dir>, or set FAH_WEIGHTS.")
SAM3D_GET=0
if [[ ! -f "$SAM3D/model_config.yaml" || ! -f "$SAM3D/assets/mhr_model.pt" ]]; then
  set +e
  ERR="$("$PY" -c "
import sys
from huggingface_hub import auth_check
from huggingface_hub.errors import RepositoryNotFoundError
try:
    auth_check('facebook/sam-3d-body-dinov3')
except RepositoryNotFoundError:  # gated: not logged in, or the licence not accepted
    sys.exit(3)
except Exception as e:
    sys.exit(f'{type(e).__name__}: {e}')
" 2>&1)"
  rc=$?
  set -e
  if ((rc == 0)); then
    SAM3D_GET=1
  elif ((rc != 3)); then
    die "Could not check access to huggingface.co/facebook/sam-3d-body-dinov3 (network?):" "$ERR"
  else
    MISSING+=("SAM 3D Body (gated, SAM License): the model config and the MHR hand model (0.7 GB, downloaded by this script).
   Accept the licence at https://huggingface.co/facebook/sam-3d-body-dinov3 (logged in), then run:
       $REPO/.venv/bin/hf auth login
   (a token with read access, from https://huggingface.co/settings/tokens).")
  fi
fi
mano_ok() { [[ -f "$1/MANO_RIGHT.pkl" && -f "$1/MANO_LEFT.pkl" ]]; }
if ! mano_ok "$MANO" && [[ -z "$MANO_SRC" ]]; then
  MISSING+=("MANO (registration, not redistributable): sign up at https://mano.is.tue.mpg.de, download \"Models & Code\"
   (mano_v1_2.zip), then run again with --mano ~/Downloads/mano_v1_2.zip (or the unpacked folder).")
fi
if ((${#MISSING[@]})); then
  printf '\nBefore the first run, %d step(s) need you:\n' "${#MISSING[@]}"
  for i in "${!MISSING[@]}"; do printf '\n%d. %s\n' $((i + 1)) "${MISSING[$i]}"; done
  printf '\nThen run %s again (with --weights / --mano if you used them).\n' "$0"
  exit 1
fi
if ((SAM3D_GET)); then
  echo "== SAM 3D Body files -> $SAM3D"
  "$PY" -c "import sys, fah; fah.download_sam3d(sys.argv[1])" "$SAM3D"
fi
if ! mano_ok "$MANO"; then
  echo "== MANO -> $MANO"
  mkdir -p "$MANO"
  if [[ -f "$MANO_SRC" ]]; then  # the zip: its two hand models, converted in place
    unzip -j -o -q "$MANO_SRC" '*/MANO_RIGHT.pkl' '*/MANO_LEFT.pkl' -d "$MANO" || die "No MANO_RIGHT.pkl / MANO_LEFT.pkl in $MANO_SRC."
    SRC="$MANO"
  else
    SRC="$(dirname "$(find "$MANO_SRC" -maxdepth 3 -name MANO_RIGHT.pkl 2>/dev/null | head -1)")"
    mano_ok "$SRC" || die "No MANO_RIGHT.pkl / MANO_LEFT.pkl in $MANO_SRC."
  fi
  "$PY" "$REPO/bench/convert_mano.py" "$SRC" "$MANO" | tail -1  # plain-numpy pickles (no chumpy)
fi

# 5. Core ML exports (once per weights directory)
EXPORTS=()  # the pose network, detector and part-free pose network
while IFS= read -r l; do EXPORTS+=("$l"); done < <("$PY" -c "import sys; from fah.api import coreml_noparts, coreml_paths
p = coreml_paths('fah-s', sys.argv[1]); print(*p, coreml_noparts(p[0]), sep='\n')" "$WEIGHTS")
if ((COREML)) && [[ ! -d "${EXPORTS[0]}" || ! -d "${EXPORTS[1]}" || ! -d "${EXPORTS[2]}" ]]; then
  echo "== Core ML exports, first run only (2-3 min) -> $WEIGHTS/coreml"
  MANO_DIR="$MANO" "$REPO/.venv/bin/fah-build-coreml" --backbone s --weights "$WEIGHTS" --sam3d "$SAM3D"
fi
if ((SETUP_ONLY)); then
  echo "Setup done. Start the camera with: $0"
  exit 0
fi

echo "== fah-live (q quits)"
set +e
MANO_DIR="$MANO" "$REPO/.venv/bin/fah-live" --backbone s --weights "$WEIGHTS" --sam3d "$SAM3D" --mano_dir "$MANO" \
  $( ((COREML)) || echo --preset torch ) ${ARGS[@]+"${ARGS[@]}"}
rc=$?
set -e
if ((rc)) && [[ " ${ARGS[*]-} " != *" --video "* ]]; then
  printf '\n%s\n' "If the camera did not open: macOS asks once per app for camera access. Accept the prompt, or enable the app you run" \
    "this in (Terminal, iTerm, VS Code, ...) in System Settings > Privacy & Security > Camera, then run $0 again" \
    "(other cameras: --camera 1)."
fi
exit $rc
