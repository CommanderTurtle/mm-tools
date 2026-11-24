#!/usr/bin/env bash
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Profile FastFrames-LingBot 1.3B on one 8-GPU node (three fresh processes per row).
#
#   LINGBOT_WORLD_ROOT=/path/to/lingbot-world-v2 bash apps/lingbot/fig5.sh outputs/fig5 [row ...]
#
# Rows: ff13_1 ff13_8 (default: both).
# PYTHON runs FastFrames (this repository's environment). CHECKPOINTS holds lingbot-1.3b/,
# and the upstream model/shared assets (see docs/installation.md#lingbot).
set -euo pipefail

usage() { sed -n '6,12p' "$0"; exit 1; }
[ $# -ge 1 ] || usage
OUT=$1; shift
ROWS=${*:-ff13_1 ff13_8}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export LINGBOT_WORLD_ROOT="${LINGBOT_WORLD_ROOT:?set LINGBOT_WORLD_ROOT to the LingBot-World clone}"
PYTHON=${PYTHON:-python}
CHECKPOINTS=${CHECKPOINTS:-checkpoints}
REPEATS=${REPEATS:-3}
DRY_RUN=${DRY_RUN:-0}
mkdir -p "${OUT}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

run() {
  echo "+ $*"
  if [ "${DRY_RUN}" != 1 ]; then "$@"; fi
}

fastframes() {  # model tag gpus repeat
  local cmd=("${PYTHON}")
  if [ "$3" -gt 1 ]; then cmd=("${PYTHON}" -m torch.distributed.run --nproc_per_node "$3"); fi
  run env -u PYTORCH_CUDA_ALLOC_CONF OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4 PYTORCH_ALLOC_CONF=backend:native \
    TORCHINDUCTOR_CACHE_DIR="${OUT}/inductor/$2_w$3" \
    "${cmd[@]}" "${REPO_ROOT}/apps/lingbot/latency.py" --model "$1" --checkpoint-dir "${CHECKPOINTS}" \
    --output "${OUT}/fastframes_$2_w$3_r$4.json"
}

for row in ${ROWS}; do
  for r in $(seq 1 "${REPEATS}"); do
    case ${row} in
      ff13_1) fastframes lingbot-1.3b 1.3b 1 "$r" ;;
      ff13_8) fastframes lingbot-1.3b 1.3b 8 "$r" ;;
      *) echo "unknown row ${row}"; usage ;;
    esac
  done
done
run "${PYTHON}" "${REPO_ROOT}/apps/lingbot/summarize.py" "${OUT}" --output "${OUT}/fig5.json"
