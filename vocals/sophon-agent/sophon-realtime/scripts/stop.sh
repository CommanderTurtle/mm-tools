#!/usr/bin/env bash
# Stop the services started by scripts/start.sh
HERE=$(cd "$(dirname "$0")/.." && pwd); LOG=${LOG_DIR:-$HERE/logs}
for n in gateway talker thinker; do
  f="$LOG/$n.pid"
  if [ -f "$f" ]; then
    p=$(cat "$f"); kill "$p" 2>/dev/null && echo "stopped $n ($p)"; pkill -P "$p" 2>/dev/null; rm -f "$f"
  fi
done
if [ -d "$LOG/mps/pipe" ]; then sleep 5; CUDA_MPS_PIPE_DIRECTORY="$LOG/mps/pipe" bash -c 'echo quit | nvidia-cuda-mps-control' 2>/dev/null && echo "stopped CUDA MPS"; fi
