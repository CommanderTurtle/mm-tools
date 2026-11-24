#!/bin/bash
# Start Sophon-Realtime-32B-Tag on ONE GPU (80 GB: H100 / H800 / A100 80G).
# usage: bash start.sh        env: GPU=0  PORT=8000  MODEL_DIR=./model  MPS=1  TRT=auto
# several GPUs: one instance per GPU, e.g. GPU=0 PORT=8000 bash start.sh; GPU=1 PORT=8001 bash start.sh  (put a load balancer in front)
# endpoints: ws://HOST:PORT/v1/realtime (full duplex)   http://HOST:PORT/v1/chat/completions (audio in -> text + audio out)
HERE=$(cd "$(dirname "$0")" && pwd); cd "$HERE"; mkdir -p logs
GPU=${GPU:-0}; PORT=${PORT:-8000}; PY=$HERE/.venv/bin; MODEL_DIR=${MODEL_DIR:-$HERE/model}
[ -f logs/server_$PORT.pid ] && PORT=$PORT bash stop.sh >/dev/null 2>&1
DEV=$GPU
case "$GPU" in *,*) MPS=0;; esac
# a 4B model (Sophon-Realtime-4B-Tag) gets the one-32-GB-GPU config unless CONFIG says otherwise
grep -q '"hidden_size": 2560' "${MODEL_DIR:-$HERE/model}/thinker/config.json" 2>/dev/null && IS4B=1
SUF=1gpu; [ -n "$IS4B" ] && SUF=4b
# the 4-bit 4B thinker (Sophon-Realtime-4B-Tag-W4) gets the one-24-GB-GPU config
[ -n "$IS4B" ] && grep -q '"quantization_config"' "${MODEL_DIR:-$HERE/model}/thinker/config.json" 2>/dev/null && SUF=4b_w4
CONFIG=${CONFIG:-sophon_realtime_tag_$SUF.yaml}
# vLLM compiles a few kernels at the first start (several minutes) and caches them. Keep the cache with this install: the default
# cache in ~/.cache is shared by every vLLM install of the user, and installs with different paths rebuild it at each start.
export FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:-$HERE}
if [ "${MPS:-1}" = 1 ] && command -v nvidia-cuda-mps-control >/dev/null; then   # the three stages share the GPU concurrently (much better first-audio under load)
  export CUDA_MPS_PIPE_DIRECTORY=$HERE/.mps/gpu$GPU/pipe CUDA_MPS_LOG_DIRECTORY=$HERE/.mps/gpu$GPU/log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1 || CUDA_VISIBLE_DEVICES=$GPU nvidia-cuda-mps-control -d
  if echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1; then DEV=0; echo "MPS on (GPU $GPU)"; else unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; echo "MPS not available; running without it"; fi
fi
TRT=${TRT:-auto}; [ "$TRT" = auto ] && { $PY/python -c "import tensorrt" 2>/dev/null && TRT=1 || TRT=0; }
[ "$TRT" = 0 ] && echo "WARNING: TensorRT is off (not installed?): the voice runs in plain PyTorch, about 3x slower under load. Re-run install.sh without NO_TRT=1."
sed "s#__HERE__#$HERE#g" config/$CONFIG > logs/deploy_$PORT.yaml
# tag-tts v1.3: distilled flow, 4 steps, no CFG; several requests' flow chunks share one TensorRT call
export SOPHON_FLOW_STEPS=${SOPHON_FLOW_STEPS:-4} SOPHON_FLOW_CFG=${SOPHON_FLOW_CFG:-0} SOPHON_FLOW_ONNX=${SOPHON_FLOW_ONNX:-$MODEL_DIR/talker/flow.decoder.estimator.fp32.onnx} COSYVOICE3_BATCH_FLOW=${COSYVOICE3_BATCH_FLOW:-1} SOPHON_FLOW_MAXB=${SOPHON_FLOW_MAXB:-$([ -n "$IS4B" ] && echo 8 || echo 16)}
SOPHON_MODEL=$MODEL_DIR SOPHON_PORT=$PORT SOPHON_BRIDGE_PORT=${SOPHON_BRIDGE_PORT:-$((PORT+700))} VLLM_DISABLE_COMPILE_CACHE=1 COSYVOICE3_TRT=$TRT CUDA_VISIBLE_DEVICES=$DEV PATH=$PY:$PATH \
  setsid nohup $PY/vllm serve $MODEL_DIR --omni --port $PORT --trust-remote-code --init-timeout 3000 --stage-init-timeout 3000 --deploy-config logs/deploy_$PORT.yaml > logs/server_$PORT.log 2>&1 < /dev/null &
echo $! > logs/server_$PORT.pid
echo "starting (first start builds kernels / TensorRT engine: 5-10 min) ... log: logs/server_$PORT.log"
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
$PY/python client_example.py --url http://127.0.0.1:$PORT --quiet >/dev/null 2>&1
echo "READY  ws://127.0.0.1:$PORT/v1/realtime   http://127.0.0.1:$PORT/v1/chat/completions   (web demo: python client_example.py)"
