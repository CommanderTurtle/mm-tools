#!/bin/bash
# Live mode: start the GPT-Live compatible gateway in front of a running Tag or Flash engine.
# usage (after `bash start.sh` in ../tag or ../flash):
#   BUILD=tag PORT=8000 LIVE_PORT=8780 bash start_live.sh
# env: BUILD=tag|flash (which install to borrow the Python environment from), PORT = engine port (tool bridge = PORT+700),
#      LIVE_PORT = gateway port, GPU = GPU for the small ASR model (default 0), SOPHON_LIVE_ASR = ASR model id or path
#      (default Qwen/Qwen3-ASR-0.6B-hf; in mainland China: modelscope download --model Qwen/Qwen3-ASR-0.6B-hf --local_dir ./asr
#       and set SOPHON_LIVE_ASR=./asr)
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; mkdir -p logs
BUILD=${BUILD:-tag}; PORT=${PORT:-8000}; LIVE_PORT=${LIVE_PORT:-8780}; GPU=${GPU:-0}
PY=$HERE/../$BUILD/.venv/bin
[ -x $PY/python ] || { echo "no Python environment at ../$BUILD/.venv: run bash install.sh in ../$BUILD first"; exit 1; }
curl -sf -m2 localhost:$PORT/health >/dev/null || { echo "engine not running on port $PORT: start it first (cd ../$BUILD && GPU=$GPU PORT=$PORT bash start.sh)"; exit 1; }
[ -f logs/live_$LIVE_PORT.pid ] && LIVE_PORT=$LIVE_PORT bash stop_live.sh >/dev/null 2>&1
CUDA_VISIBLE_DEVICES=$GPU setsid nohup $PY/python -m sophon_live.gateway --port $LIVE_PORT \
  --engine ws://127.0.0.1:$PORT/v1/realtime --bridge http://127.0.0.1:$((PORT+700)) > logs/live_$LIVE_PORT.log 2>&1 < /dev/null &
echo $! > logs/live_$LIVE_PORT.pid
for i in $(seq 1 120); do grep -q "Sophon Live gateway on" logs/live_$LIVE_PORT.log && break
  kill -0 $(cat logs/live_$LIVE_PORT.pid) 2>/dev/null || { echo "gateway exited — see logs/live_$LIVE_PORT.log"; tail -20 logs/live_$LIVE_PORT.log; exit 1; }; sleep 2; done
grep -q "Sophon Live gateway on" logs/live_$LIVE_PORT.log || { echo "timeout — see logs/live_$LIVE_PORT.log"; exit 1; }
echo "READY  GPT-Live: ws://127.0.0.1:$LIVE_PORT/v1/live/sessions   test page: http://127.0.0.1:$LIVE_PORT/"
