#!/usr/bin/env bash
# One SFT Arrow in, one self-contained SFT Arrow out. Additional CLI flags pass through.
# VAE_PATH=/path/to/dcae-v.pt NUM_GPUS=8 bash "$0" input.arrow output.arrow
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPTS_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

INPUT_ARROW="${INPUT_ARROW:-}"
OUTPUT_ARROW="${OUTPUT_ARROW:-}"
if [[ $# -gt 0 && "$1" != --* ]]; then
    INPUT_ARROW="$1"
    shift
fi
if [[ $# -gt 0 && "$1" != --* ]]; then
    OUTPUT_ARROW="$1"
    shift
fi
if [[ -z "${INPUT_ARROW}" ]]; then
    echo "Usage: $0 input.arrow [output.arrow] [--rows 0,3 | other Python options]" >&2
    exit 2
fi
OUTPUT_ARROW="${OUTPUT_ARROW:-${INPUT_ARROW%.arrow}_vae.arrow}"
if [[ "${HOST_NUM:-1}" != 1 ]]; then
    echo "This single-writer launcher supports one node only; set HOST_NUM=1." >&2
    exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1

flags=(
    --input-arrow "${INPUT_ARROW}"
    --output-arrow "${OUTPUT_ARROW}"
    --device "${DEVICE:-cuda}"
    --num-gpus "${NUM_GPUS:-${HOST_GPU_NUM:-8}}"
    --vae-precision "${VAE_PRECISION:-bf16}"
    --fps 24
    --boundary-fps "${BOUNDARY_FPS:-24}"
    --height "${HEIGHT:-704}"
    --width "${WIDTH:-1280}"
    --prefetch "${PREFETCH:-1}"
)
if [[ -n "${VAE_PATH:-}" ]]; then
    flags+=(--vae-checkpoint "${VAE_PATH}")
fi
if [[ -n "${GPU_IDS:-}" ]]; then
    flags+=(--gpu-ids "${GPU_IDS}")
fi

exec "${PYTHON_BIN}" "${SCRIPTS_DIR}/vae_latent_encoding.py" "${flags[@]}" "$@"
