#!/usr/bin/env bash
# Start the Sophon-Realtime E2E build on ONE GPU: thinker (vLLM) + speaking engine + gateway with the web demo.
# usage: SOPHON_MODEL_DIR=/path/to/Sophon-Realtime-32B-E2E [GPU=0 PORT=8000 MPS=1 SLOTS=16] bash scripts/start.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)
: "${SOPHON_MODEL_DIR:?set SOPHON_MODEL_DIR to the downloaded Sophon-Realtime-32B-E2E directory}"
[ -d "$SOPHON_MODEL_DIR/talker220" ] || { echo "No talker220/ in $SOPHON_MODEL_DIR: these are the weights of an older (two-GPU) release, download the current ones."; exit 1; }
GPU=${GPU:-0}; PORT=${PORT:-8000}
THINKER_PORT=${THINKER_PORT:-8300}; TALKER_PORT=${TALKER_PORT:-8120}
THINKER_ENV=${THINKER_ENV:-$HERE/.venv-thinker}; ENGINE_ENV=${ENGINE_ENV:-$HERE/.venv-engine}
REPLY_LORA=${SOPHON_REPLY_LORA:-$SOPHON_MODEL_DIR/adapters/tag}
LORA_RANK=${LORA_RANK:-16}
LOG=${LOG_DIR:-$HERE/logs}; mkdir -p "$LOG"
export SOPHON_MODEL_DIR PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

# The thinker and the speaking engine are two processes on one GPU. With CUDA MPS they use it at the same time;
# without it they take turns, which costs latency and concurrency (first audio 0.6 s instead of 0.25 s for one user).
DEV=$GPU
if [ "${MPS:-1}" = 1 ] && command -v nvidia-cuda-mps-control > /dev/null; then
  export CUDA_MPS_PIPE_DIRECTORY="$LOG/mps/pipe" CUDA_MPS_LOG_DIRECTORY="$LOG/mps/log"; mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  echo get_server_list | nvidia-cuda-mps-control > /dev/null 2>&1 || CUDA_VISIBLE_DEVICES=$GPU nvidia-cuda-mps-control -d
  sleep 1
  if echo get_server_list | nvidia-cuda-mps-control > /dev/null 2>&1; then DEV=0; echo "CUDA MPS on (GPU $GPU)"
  else unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; echo "WARNING: CUDA MPS could not be started; running without it (slower under load)"; fi
else
  echo "WARNING: CUDA MPS is off; the thinker and the speaking engine will time-slice the GPU (slower under load)"
fi

echo "[1/3] thinker on GPU $GPU (port $THINKER_PORT)"
# vLLM compiles a few kernels at the first start (several minutes) and caches them. Keep the cache with this install: the default
# cache in ~/.cache is shared by every vLLM install of the user, and installs with different paths rebuild it at each start.
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$HERE}
CUDA_VISIBLE_DEVICES=$DEV PATH="$THINKER_ENV/bin:$PATH" nohup "$THINKER_ENV/bin/vllm" serve "$SOPHON_MODEL_DIR/base" \
  --served-model-name dj --enable-lora --lora-modules sophon="$REPLY_LORA" --max-lora-rank "$LORA_RANK" \
  --quantization fp8 --max-model-len 8192 --gpu-memory-utilization ${THINKER_MEM:-0.55} --port "$THINKER_PORT" \
  > "$LOG/thinker.log" 2>&1 < /dev/null & echo $! > "$LOG/thinker.pid"

wait_for() {  # url, name, pid file
  for i in $(seq 1 180); do
    if curl -sf "$1" > /dev/null 2>&1; then echo "  $2 ready"; return 0; fi
    if ! kill -0 "$(cat "$3")" 2>/dev/null; then echo "  $2 exited, see ${3%.pid}.log"; exit 1; fi
    sleep 5
  done; echo "  $2 not ready after 15 min, see ${3%.pid}.log"; exit 1
}
wait_for "http://127.0.0.1:$THINKER_PORT/v1/models" thinker "$LOG/thinker.pid"

echo "[2/3] speaking engine on GPU $GPU (port $TALKER_PORT)"
CUDA_VISIBLE_DEVICES=$DEV S220_SLOTS=${SLOTS:-16} OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTHONUNBUFFERED=1 nohup "$ENGINE_ENV/bin/python" -m sophon_expressive.talker220_server \
  --port "$TALKER_PORT" > "$LOG/talker.log" 2>&1 < /dev/null & echo $! > "$LOG/talker.pid"
wait_for "http://127.0.0.1:$TALKER_PORT/health" "speaking engine" "$LOG/talker.pid"

echo "[3/3] gateway + web demo on port $PORT"
PYTHONUNBUFFERED=1 nohup "${GATEWAY_ENV:-$ENGINE_ENV}/bin/python" -m sophon_expressive.gateway --port "$PORT" \
  --dj "http://127.0.0.1:$THINKER_PORT" --talker32 "http://127.0.0.1:$TALKER_PORT" > "$LOG/gateway.log" 2>&1 < /dev/null & echo $! > "$LOG/gateway.pid"
sleep 3
echo "Open http://<this machine>:$PORT/  (WebSocket: ws://<this machine>:$PORT/ws)"
