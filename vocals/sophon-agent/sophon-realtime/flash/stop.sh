#!/bin/bash
# stop one instance (PORT=8000 bash stop.sh) or all instances started from this folder (bash stop.sh)
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"
for f in logs/server_${PORT:-*}.pid; do
  [ -f "$f" ] || continue
  kill -- -$(ps -o pgid= -p $(cat $f) | tr -d " ") 2>/dev/null; rm -f $f
done
sleep 5
if [ -z "$PORT" ]; then for d in .mps/gpu*/pipe; do [ -d "$d" ] && CUDA_MPS_PIPE_DIRECTORY=$HERE/$d bash -c 'echo quit | nvidia-cuda-mps-control' 2>/dev/null; done; fi
echo stopped
