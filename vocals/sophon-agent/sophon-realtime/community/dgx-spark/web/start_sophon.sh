#!/usr/bin/env bash
# ============================================================
# 启动 Sophon-Realtime 后端（进程 A：加载权重 + 跑推理，监听 8000）
#
# 配套：网页前端另开终端跑 python ~/sophon_web.py（进程 B）
# 顺序：先跑本脚本，等出现 "Application startup complete" 再起前端
#
# 用法：
#   bash ~/start_sophon.sh                       # 默认参数，前台运行
#   bash ~/start_sophon.sh --port 8001           # 换端口
#   bash ~/start_sophon.sh --daemon              # 后台运行，控制台日志写 ~/sophon/logs/
#   SOPHON_GPU=0 bash ~/start_sophon.sh          # 也可用环境变量覆盖
# ============================================================

set -euo pipefail

# ---- 默认配置（命令行参数 > 环境变量 > 这里的默认值）----
MODEL="${SOPHON_MODEL:-$HOME/data/models/Sophon-Realtime-32B-Tag}"
HOST="${SOPHON_HOST:-0.0.0.0}"
PORT="${SOPHON_PORT:-8000}"
GPU="${SOPHON_GPU:-0}"
LOG_DIR="${SOPHON_LOG_DIR:-$HOME/sophon/logs}"
VENV="${SOPHON_VENV:-$HOME/sophon/venv}"
DAEMON=0

usage(){
  cat <<'EOF'
用法：bash start_sophon.sh [选项]

  --model PATH      模型目录（默认 ~/data/models/Sophon-Realtime-32B-Tag）
  --host ADDR       监听地址（默认 0.0.0.0）
  --port N          端口（默认 8000）
  --gpu N           GPU 编号（默认 0）
  --log-dir DIR     日志目录（默认 ~/sophon/logs）
  --daemon          后台运行，控制台日志写到日志目录
  -h, --help        显示本帮助

也可用环境变量：SOPHON_MODEL / SOPHON_HOST / SOPHON_PORT /
                SOPHON_GPU / SOPHON_LOG_DIR / SOPHON_VENV
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   MODEL="${2:?--model 需要一个值}"; shift 2 ;;
    --host)    HOST="${2:?--host 需要一个值}"; shift 2 ;;
    --port)    PORT="${2:?--port 需要一个值}"; shift 2 ;;
    --gpu)     GPU="${2:?--gpu 需要一个值}"; shift 2 ;;
    --log-dir) LOG_DIR="${2:?--log-dir 需要一个值}"; shift 2 ;;
    --daemon)  DAEMON=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1（用 --help 看用法）" >&2; exit 1 ;;
  esac
done

# ---- 前置检查：避免等 10 分钟才发现路径写错 ----
if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "错误：找不到虚拟环境 $VENV/bin/activate" >&2
  echo "先建环境：python3.12 -m venv $VENV" >&2
  exit 1
fi
if [[ ! -d "$MODEL" ]]; then
  echo "错误：找不到模型目录 $MODEL" >&2
  echo "先下载：modelscope download --model AdventistsAI/Sophon-Realtime-32B-Tag --local_dir $MODEL" >&2
  exit 1
fi
mkdir -p "$LOG_DIR"

# 端口占用只做警告，不阻断（有些环境 ss 不可用，忽略即可）
if command -v ss >/dev/null 2>&1 \
   && ss -ltn 2>/dev/null | grep -qE ":${PORT}([[:space:]]|$)"; then
  echo "警告：端口 $PORT 已被监听，可能已有实例在跑" >&2
fi

# ---- 环境 ----
# shellcheck disable=SC1090
source "$VENV/bin/activate"
export FLASHINFER_WORKSPACE_BASE="$HOME/sophon/venv"
cd "$(dirname "$VENV")"      # 回到 ~/sophon，与手动启动时的工作目录一致

CMD=(
  sophon-realtime serve
  --model   "$MODEL"
  --host    "$HOST"
  --port    "$PORT"
  --gpu     "$GPU"
  --log-dir "$LOG_DIR"
)

echo "============================================================"
echo "后端：$HOST:$PORT    模型：$MODEL    GPU：$GPU"
echo "启动：${CMD[*]}"
echo "首次加载约 10 分钟，出现 'Application startup complete' 即为就绪"
echo "============================================================"

if [[ $DAEMON -eq 1 ]]; then
  STAMP="$(date +%Y%m%d-%H%M%S)"
  OUT_LOG="$LOG_DIR/console-$STAMP.log"
  nohup "${CMD[@]}" > "$OUT_LOG" 2>&1 &
  PID=$!
  echo "已后台启动：PID=$PID"
  echo "控制台日志：$OUT_LOG"
  echo "跟踪日志：tail -f $OUT_LOG"
  echo "停止：kill $PID"
else
  exec "${CMD[@]}"
fi
