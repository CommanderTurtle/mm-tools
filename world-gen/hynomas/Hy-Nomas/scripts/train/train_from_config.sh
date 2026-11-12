#!/usr/bin/env bash
# Config-first launcher for local and multi-node training.
#
# Run this same command on every node; the launcher reads HOST_NUM,
# HOST_GPU_NUM, INDEX, and CHIEF_IP:
#
#   bash scripts/train/train_from_config.sh \
#     configs/experiment/sft.yaml
#
# Any remaining arguments are forwarded as Hydra overrides.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <experiment.yaml|experiment-name> [Hydra overrides...]" >&2
    exit 2
fi

EXPERIMENT=$1
shift

# Hydra expects a path relative to configs/experiment, without .yaml.
EXPERIMENT=${EXPERIMENT%.yaml}
EXPERIMENT=${EXPERIMENT#configs/experiment/}

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

NNODES=${HOST_NUM:-1}
NPROC_PER_NODE=${HOST_GPU_NUM:-${NPROC_PER_NODE:-8}}
NODE_RANK=${INDEX:-0}

if [[ -n "${CHIEF_IP:-}" ]]; then
    MASTER_ADDR=${CHIEF_IP%%:*}
elif [[ -n "${NODE_IP_LIST:-}" ]]; then
    MASTER_ADDR=$(printf '%s' "$NODE_IP_LIST" | tr ',' '\n' | head -n 1)
    MASTER_ADDR=${MASTER_ADDR%%:*}
else
    MASTER_ADDR=127.0.0.1
fi

MASTER_PORT=${RDZV_PORT:-29500}
RDZV_ID_VALUE=${RDZV_ID:-789}

echo "=========================================="
echo "Hy-Nomas Training Launcher"
echo "=========================================="
echo "Experiment:       $EXPERIMENT"
echo "Nodes:            $NNODES"
echo "GPUs per node:    $NPROC_PER_NODE"
echo "Node rank:        $NODE_RANK"
echo "Rendezvous:       ${MASTER_ADDR}:${MASTER_PORT}"
echo "Rendezvous ID:    $RDZV_ID_VALUE"
echo "Hydra overrides:  $*"
echo "=========================================="

torchrun \
    --nnodes "$NNODES" \
    --nproc_per_node "$NPROC_PER_NODE" \
    --node_rank "$NODE_RANK" \
    --rdzv_endpoint "${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv_id "$RDZV_ID_VALUE" \
    hymm/train.py \
    "+experiment=${EXPERIMENT}" \
    "$@"
