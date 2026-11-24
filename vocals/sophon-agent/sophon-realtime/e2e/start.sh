#!/bin/bash
# Start Sophon-Realtime-32B-E2E (vLLM-Omni, three stages) on ONE GPU.   usage: bash start.sh   env: GPU=0 PORT=8000 MPS=1 SLOTS=16 MODEL_DIR=./model
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; mkdir -p logs
GPU=${GPU:-0}; PORT=${PORT:-8000}; PY=${PY:-$HERE/.venv/bin}; MODEL_DIR=${MODEL_DIR:-$HERE/model}
[ -f "$MODEL_DIR/talker220/config.json" ] || { echo "no model at $MODEL_DIR (need thinker/ talker220/ e2/ voices/)"; exit 1; }
[ -f logs/server_$PORT.pid ] && PORT=$PORT bash stop.sh >/dev/null 2>&1
DEV=$GPU
# a 4B model (Sophon-Realtime-4B-E2E) gets the one-32-GB-GPU config unless CONFIG says otherwise
grep -q '"hidden_size": 2560' "${MODEL_DIR:-$HERE/model}/thinker/config.json" 2>/dev/null && IS4B=1
SUF=1gpu; [ -n "$IS4B" ] && SUF=4b
CONFIG=${CONFIG:-sophon_realtime_e2e_$SUF.yaml}
# vLLM compiles a few kernels at the first start (several minutes) and caches them. Keep the cache with this install: the default
# cache in ~/.cache is shared by every vLLM install of the user, and installs with different paths rebuild it at each start.
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$HERE}
if [ "${MPS:-1}" = 1 ] && command -v nvidia-cuda-mps-control >/dev/null; then   # the three stages share the GPU concurrently
  export CUDA_MPS_PIPE_DIRECTORY=$HERE/.mps/gpu$GPU/pipe CUDA_MPS_LOG_DIRECTORY=$HERE/.mps/gpu$GPU/log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1 || CUDA_VISIBLE_DEVICES=$GPU nvidia-cuda-mps-control -d
  if echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then DEV=0; echo "MPS on (GPU $GPU)"; else unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; echo "MPS not available; running without it"; fi
fi
sed "s#__HERE__#$HERE#g" config/$CONFIG > logs/deploy_$PORT.yaml
SOPHON_MODEL=$MODEL_DIR SOPHON_PORT=$PORT SOPHON_BRIDGE_PORT=${SOPHON_BRIDGE_PORT:-$((PORT+700))} SOPHON_E2E_SLOTS=${SLOTS:-$([ -n "$IS4B" ] && echo 8 || echo 16)} SOPHON_E2E_DEC_MAXB=${SOPHON_E2E_DEC_MAXB:-$([ -n "$IS4B" ] && echo 8 || echo 16)} OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} VLLM_DISABLE_COMPILE_CACHE=1 CUDA_VISIBLE_DEVICES=$DEV PATH=$PY:$PATH \
  setsid nohup $PY/vllm serve $MODEL_DIR --omni --port $PORT --trust-remote-code --init-timeout 3000 --stage-init-timeout 3000 --deploy-config logs/deploy_$PORT.yaml > logs/server_$PORT.log 2>&1 < /dev/null &
echo $! > logs/server_$PORT.pid
echo "starting (about 5 min) ... log: logs/server_$PORT.log"
for i in $(seq 1 240); do
  curl -sf -m2 localhost:$PORT/health >/dev/null && break
  kill -0 $(cat logs/server_$PORT.pid) 2>/dev/null || { echo "server exited — see logs/server_$PORT.log"; tail -20 logs/server_$PORT.log; exit 1; }
  sleep 5
done
curl -sf -m2 localhost:$PORT/health >/dev/null || { echo "timeout — see logs/server_$PORT.log"; exit 1; }
echo "warming up ..."
# After it starts answering, the server warms itself up in the background (a throwaway realtime session, about 20 s). Wait for that to
# finish, so that the first real request is not slow.   SOPHON_WARMUP=0 turns the background warm-up off.
if [ "${SOPHON_WARMUP:-1}" != 0 ]; then
  for i in $(seq 1 90); do grep -aq "realtime warm-up \([0-9] got a reply\|1 got no reply\|1 failed\)" logs/server_$PORT.log && break; sleep 2; done
fi
for k in 1 2; do $PY/python client_example.py --url http://127.0.0.1:$PORT --quiet --out /dev/null >/dev/null 2>&1; done
echo "READY  ws://127.0.0.1:$PORT/v1/realtime   http://127.0.0.1:$PORT/v1/chat/completions"
