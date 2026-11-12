#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
RUNTIME_DIR="${RUNTIME_DIR:-${PWD}/.runtime/seg_ar_demo}"
MPS_PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-${RUNTIME_DIR}/mps-pipe}"
MPS_LOG_DIR="${CUDA_MPS_LOG_DIRECTORY:-${RUNTIME_DIR}/mps-log}"

descendant_pids() {
  local root_pid="$1"
  python - "${root_pid}" <<'PY'
import pathlib
import sys

pending = [int(sys.argv[1])]
seen = set()
while pending:
    parent = pending.pop()
    try:
        children = pathlib.Path(
            f"/proc/{parent}/task/{parent}/children"
        ).read_text().split()
    except OSError:
        continue
    for raw_child in children:
        child = int(raw_child)
        if child not in seen:
            seen.add(child)
            pending.append(child)
print(" ".join(str(pid) for pid in sorted(seen)))
PY
}

pid_files=(
  "${RUNTIME_DIR}/http_server_8081.pid"
  "${RUNTIME_DIR}/http_server_8081.launcher.pid"
  "${RUNTIME_DIR}/auth_proxy.pid"
  "${RUNTIME_DIR}/dispatcher.pid"
  "${RUNTIME_DIR}/gateway.pid"
  "${RUNTIME_DIR}/gateway.launcher.pid"
)
if [[ -f "${RUNTIME_DIR}/worker.pids" ]]; then
  while IFS= read -r worker_pid; do
    [[ -n "${worker_pid}" ]] || continue
    worker_pid_file="${RUNTIME_DIR}/worker_${worker_pid}.pid"
    printf '%s\n' "${worker_pid}" > "${worker_pid_file}"
    pid_files+=("${worker_pid_file}")
  done < "${RUNTIME_DIR}/worker.pids"
elif [[ -f "${RUNTIME_DIR}/worker.pid" ]]; then
  pid_files+=("${RUNTIME_DIR}/worker.pid")
fi

for pid_file in "${pid_files[@]}"; do
  name="$(basename "${pid_file}" .pid)"
  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: no pid file"
    continue
  fi
  pid="$(cat "${pid_file}")"
  if kill -0 "${pid}" 2>/dev/null; then
    descendants="$(descendant_pids "${pid}")"
    targets="${pid} ${descendants}"
    pgids="$(
      for target in ${targets}; do
        ps -o pgid= -p "${target}" 2>/dev/null || true
      done | tr -d ' ' | sort -u
    )"
    for pgid in ${pgids}; do
      kill -TERM -- "-${pgid}" 2>/dev/null || true
    done
    echo "${name}: sent SIGTERM to process tree for ${pid}"
    for _ in $(seq 1 40); do
      alive=0
      for target in ${targets}; do
        if kill -0 "${target}" 2>/dev/null; then
          alive=1
          break
        fi
      done
      if [[ "${alive}" == "0" ]]; then
        break
      fi
      sleep 0.25
    done
    for target in ${targets}; do
      if kill -0 "${target}" 2>/dev/null; then
        target_pgid="$(ps -o pgid= -p "${target}" | tr -d ' ')"
        kill -KILL -- "-${target_pgid}" 2>/dev/null \
          || kill -KILL "${target}" 2>/dev/null \
          || true
        echo "${name}: sent SIGKILL after timeout to ${target}"
      fi
    done
  else
    echo "${name}: process ${pid} is not running"
  fi
  rm -f "${pid_file}"
done
rm -f "${RUNTIME_DIR}/worker.pids" "${RUNTIME_DIR}/worker.pid"

# The pool normally shuts down its private CUDA MPS daemon from its EXIT trap.
# Also close the runtime-local control socket here so an abrupt worker-pool
# termination cannot leave a daemon behind. A non-default shared MPS path is
# never guessed or touched.
if [[ -S "${MPS_PIPE_DIR}/control" ]] \
    && command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  timeout 10s env \
    CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
    bash -c 'printf "quit\n" | nvidia-cuda-mps-control' \
    >/dev/null 2>&1 || true
  for _ in $(seq 1 40); do
    [[ ! -S "${MPS_PIPE_DIR}/control" ]] && break
    sleep 0.25
  done
  echo "cuda-mps: requested shutdown for ${MPS_PIPE_DIR}"
fi
rm -rf "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
