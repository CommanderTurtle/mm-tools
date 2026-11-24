#!/bin/bash
# Sophon Realtime for Mac (Apple Silicon, MLX) -- one-command install.
#   bash install.sh            # auto-detects whether Hugging Face is reachable; otherwise uses ModelScope + mirrors (mainland China)
#   SOPHON_HOME=/path bash install.sh   # install somewhere other than ~/sophon-mac
#   SOPHON_SRC=hf|ms bash install.sh    # force the weight source
#   SIZE=4B bash install.sh             # the 4B model (16 GB Macs; faster, less knowledgeable). Default: 32B
set -euo pipefail
H="${SOPHON_HOME:-$HOME/sophon-mac}"; HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIZE="${SIZE:-32B}"
case "$SIZE" in
  32B|32b) SIZE=32B; REPO="Sophon-Realtime-32B-Flash-4bit-MLX"; MIN_GB=32; REC_GB=36; DISK_GB=30; USE_GB=26; DL_GB=24 ;;
  4B|4b)   SIZE=4B;  REPO="Sophon-Realtime-4B-Flash-MLX";       MIN_GB=16; REC_GB=24; DISK_GB=12; USE_GB=11; DL_GB=9 ;;
  *) echo "SIZE must be 32B or 4B" >&2; exit 1 ;;
esac
REPO_HF="adventists-ai/$REPO"; REPO_MS="AdventistsAI/$REPO"
M="$H/models"; [ "$SIZE" = 4B ] && M="$H/models-4b"   # each size has its own model folder, so both can be installed side by side
say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

# --- checks -------------------------------------------------------------------------------------------------------------------------
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "needs a Mac with Apple silicon (M1 or later)."
MAJ=$(sw_vers -productVersion | cut -d. -f1); [ "$MAJ" -ge 14 ] || die "needs macOS 14 or later."
MEM=$(( $(sysctl -n hw.memsize) / 1073741824 ))
if [ "$MEM" -lt "$MIN_GB" ]; then
  [ "$SIZE" = 32B ] && [ "$MEM" -ge 16 ] && die "the 32B model needs at least 32 GB of unified memory (this Mac has ${MEM} GB). Try the 4B model: SIZE=4B bash mac/install.sh"
  die "needs at least ${MIN_GB} GB of unified memory (this Mac has ${MEM} GB). The $SIZE model uses about ${USE_GB} GB while talking."
fi
[ "$MEM" -ge "$REC_GB" ] || echo "Note: ${MEM} GB works but is tight; close other apps before starting. ${REC_GB} GB or more is recommended."
mkdir -p "$H"; FREE=$(df -g "$H" | awk 'NR==2{print $4}')
[ "$FREE" -ge "$DISK_GB" ] || die "needs about ${DISK_GB} GB of free disk (the $SIZE model is ~${DL_GB} GB); ${FREE} GB free at $H."

SRC="${SOPHON_SRC:-}"
if [ -z "$SRC" ]; then curl -s -m 6 -o /dev/null https://huggingface.co && SRC=hf || SRC=ms; fi
if [ "$SRC" = ms ]; then
  export UV_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple" PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
  echo "Using ModelScope and the Tsinghua PyPI mirror."
fi

# --- python environment ----------------------------------------------------------------------------------------------------------------
say "Python environment"
UV="$(command -v uv || true)"
if [ -z "$UV" ]; then
  if /usr/bin/python3 -c 'import sys; sys.exit(0)' >/dev/null 2>&1; then
    /usr/bin/python3 -m venv "$H/.boot" && "$H/.boot/bin/pip" install -q uv && UV="$H/.boot/bin/uv"
  else
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$H/.boot/bin" INSTALLER_NO_MODIFY_PATH=1 sh && UV="$H/.boot/bin/uv"
  fi
fi
[ -x "$UV" ] || die "could not install uv."
[ -x "$H/.venv/bin/python" ] || "$UV" venv -q --python 3.12 "$H/.venv"
"$UV" pip install -q --python "$H/.venv/bin/python" "mlx>=0.29" "mlx-lm>=0.28" "mlx-audio==0.6.5" livekit sounddevice soundfile scipy numpy websockets \
  "transformers>=4.57" huggingface_hub modelscope

# --- code ------------------------------------------------------------------------------------------------------------------------------
say "Program files"
mkdir -p "$H/app"
for f in engine.py mlx_ear.py aec.py live.py serve.py speaker.py system_prompt.json; do rm -f "$H/app/$f"; cp "$HERE/$f" "$H/app/$f"; chmod 644 "$H/app/$f"; done
L=""; [ "$SIZE" = 4B ] && L="-4b"   # 32B keeps the old command names; 4B gets sophon-4b / sophon-4b-serve
cat > "$H/sophon$L" <<EOS
#!/bin/bash
export SOPHON_MODELS="$M"
exec "$H/.venv/bin/python" "$H/app/live.py" "\$@"
EOS
cat > "$H/sophon$L-serve" <<EOS
#!/bin/bash
export SOPHON_MODELS="$M"
exec "$H/.venv/bin/python" "$H/app/serve.py" "\$@"
EOS
chmod +x "$H/sophon$L" "$H/sophon$L-serve"

# --- weights ---------------------------------------------------------------------------------------------------------------------------
# 32B: the Flash-4bit weights repacked for MLX (bit-identical). 4B: the Flash thinker quantized by mlx_lm.
# convert.py builds either from the CUDA repo.
if [ ! -f "$M/thinker_mlx/model.safetensors.index.json" ] && [ ! -f "$M/thinker_mlx/model.safetensors" ] || [ ! -f "$M/.complete" ]; then
  say "Downloading the $SIZE model (~${DL_GB} GB, once; run again to resume if interrupted)"
  mkdir -p "$M"
  if [ "$SRC" = hf ]; then
    "$H/.venv/bin/hf" download "$REPO_HF" --local-dir "$M" || die "download failed; run the installer again to resume."
  else
    "$H/.venv/bin/modelscope" download --model "$REPO_MS" --local_dir "$M" || die "download failed; run the installer again to resume."
  fi
  touch "$M/.complete"
fi

say "Done."
echo "Start talking:   $H/sophon$L            (Chinese)"
echo "                 $H/sophon$L --lang en  (English)"
echo "A microphone is needed (a Mac Studio / mini has none built in). Echo cancellation is on, so speakers are fine; with a headset you can add --no-aec."
