#!/bin/bash
# Stop Sophon-Realtime-32B-E2E started by start.sh.   env: PORT=8000 GPU=0
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; PORT=${PORT:-8000}; GPU=${GPU:-0}
if [ -f logs/server_$PORT.pid ]; then
  P=$(cat logs/server_$PORT.pid); kill $P 2>/dev/null
  for i in $(seq 1 60); do kill -0 $P 2>/dev/null || break; sleep 1; done
  kill -9 -- -$(ps -o pgid= -p $P 2>/dev/null | tr -d ' ') 2>/dev/null; rm -f logs/server_$PORT.pid; echo "stopped server ($P)"
fi
if [ -d .mps/gpu$GPU/pipe ] && ! ls logs/server_*.pid >/dev/null 2>&1; then
  CUDA_MPS_PIPE_DIRECTORY=$HERE/.mps/gpu$GPU/pipe CUDA_MPS_LOG_DIRECTORY=$HERE/.mps/gpu$GPU/log sh -c 'echo quit | nvidia-cuda-mps-control' >/dev/null 2>&1 && echo "stopped CUDA MPS"
fi
