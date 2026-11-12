#!/usr/bin/env bash
# Run one SGLang Kimi-K3 rank. Launch the same command on every node with a
# distinct K3_NODE_RANK and a shared K3_DIST_INIT_ADDR for multi-node serving.
set -euo pipefail

: "${K3_MODEL_PATH:?Set K3_MODEL_PATH to the local Kimi-K3 checkpoint}"
K3_NNODES="${K3_NNODES:-4}"
K3_NODE_RANK="${K3_NODE_RANK:-0}"
K3_TP_SIZE="${K3_TP_SIZE:-8}"
K3_PP_SIZE="${K3_PP_SIZE:-4}"
K3_DP_SIZE="${K3_DP_SIZE:-1}"
K3_EP_SIZE="${K3_EP_SIZE:-8}"
if (( K3_NNODES > 1 )); then
  : "${K3_DIST_INIT_ADDR:?Set K3_DIST_INIT_ADDR to a reachable rank-zero host:port}"
fi
K3_DIST_INIT_ADDR="${K3_DIST_INIT_ADDR:-127.0.0.1:50000}"
K3_HOST="${K3_HOST:-0.0.0.0}"
K3_PORT="${K3_PORT:-30000}"
if (( K3_PP_SIZE == 4 )); then
  export SGLANG_PP_LAYER_PARTITION="${SGLANG_PP_LAYER_PARTITION:-24,23,23,23}"
fi

args=(
  --trust-remote-code
  --model-path "${K3_MODEL_PATH}"
  --served-model-name "${K3_SERVED_MODEL_NAME:-kimi-k3}"
  --tp-size "${K3_TP_SIZE}"
  --pp-size "${K3_PP_SIZE}"
  --dp-size "${K3_DP_SIZE}"
  --nnodes "${K3_NNODES}"
  --node-rank "${K3_NODE_RANK}"
  --dist-init-addr "${K3_DIST_INIT_ADDR}"
  --ep-size "${K3_EP_SIZE}"
  --mem-fraction-static "${K3_MEM_FRACTION_STATIC:-0.90}"
  --context-length "${K3_CONTEXT_LENGTH:-262144}"
  --mm-feature-transport cpu
  --mm-processor-worker-num "${K3_MM_PROCESSOR_WORKERS:-2}"
  --mm-io-worker-num "${K3_MM_IO_WORKERS:-16}"
  --dist-timeout "${K3_DIST_TIMEOUT:-3600}"
  --reasoning-parser kimi_k3
  --tool-call-parser kimi_k3
  --host "${K3_HOST}"
  --port "${K3_PORT}"
  --max-running-requests "${K3_MAX_RUNNING_REQUESTS:-144}"
  --cuda-graph-max-bs-decode "${K3_CUDA_GRAPH_MAX_BS_DECODE:-64}"
  --mamba-full-memory-ratio "${K3_MAMBA_FULL_MEMORY_RATIO:-1.1}"
  --chunked-prefill-size "${K3_CHUNKED_PREFILL_SIZE:-16384}"
)
if (( K3_PP_SIZE > 1 )); then
  args+=(
    --pp-max-micro-batch-size "${K3_PP_MAX_MICRO_BATCH_SIZE:-32}"
    --pp-async-batch-depth "${K3_PP_ASYNC_BATCH_DEPTH:-8}"
  )
fi
if [[ -n "${K3_API_KEY:-}" ]]; then args+=(--api-key "${K3_API_KEY}"); fi
if [[ "${K3_MOE_RUNNER_BACKEND:-marlin}" != "none" ]]; then args+=(--moe-runner-backend "${K3_MOE_RUNNER_BACKEND:-marlin}"); fi
if [[ "${K3_DECODE_ATTENTION_BACKEND:-flashmla}" != "none" ]]; then args+=(--decode-attention-backend "${K3_DECODE_ATTENTION_BACKEND:-flashmla}"); fi
if [[ -n "${K3_EXTRA_ARGS:-}" ]]; then
  read -r -a extra_args <<< "${K3_EXTRA_ARGS}"
  args+=("${extra_args[@]}")
fi
exec python3 -m sglang.launch_server "${args[@]}" "$@"
