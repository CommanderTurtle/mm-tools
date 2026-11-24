# SPDX-License-Identifier: AGPL-3.0-only
# Source from a studio after setting STUDIO to its absolute directory.
: "${STUDIO:?studio path required}"
export VIRTUAL_ENV="$STUDIO/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export GRADIO_ANALYTICS_ENABLED=False HF_HUB_DISABLE_TELEMETRY=1 DO_NOT_TRACK=1
export WANDB_MODE=disabled OTEL_SDK_DISABLED=true UV_NO_PROGRESS=1
cd "$STUDIO"

# Install source only at setup; never reset a user's existing checkout.
source_pin() {
  local dest="$1" url="$2" rev="$3"
  if [[ ! -e "$dest" ]]; then
    mkdir -p "$(dirname "$dest")"
    local tmp; tmp="$(mktemp -d "${dest}.XXXXXX")"
    git -C "$tmp" init -q && git -C "$tmp" remote add origin "$url" &&
      git -C "$tmp" fetch -q --depth=1 origin "$rev" &&
      git -C "$tmp" checkout -q --detach FETCH_HEAD && mv -T "$tmp" "$dest" || return 1
  fi
  [[ "$(git -C "$dest" rev-parse HEAD)" == "$rev" ]] || {
    printf 'Expected %s at %s\n' "$rev" "$dest" >&2; return 1;
  }
}

studio_setup() {
  [[ -x "$VIRTUAL_ENV/bin/python" ]] || uv venv --seed --python "$1" "$VIRTUAL_ENV"
}
studio_start() {
  [[ -x "$VIRTUAL_ENV/bin/python" ]] || { echo 'Run ./setupwithuv.sh first.' >&2; return 1; }
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
  exec "$VIRTUAL_ENV/bin/python" "$@"
}
