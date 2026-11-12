#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

HOSTS_FILE="${HOSTS_FILE:-configs/seg_ar_demo_hosts.json}"
RESOURCE_DB="${RESOURCE_DB:-/tmp/nomas_seg_ar_resources.json}"
PORT="${PORT:-8080}"
LEASE_SECONDS="${LEASE_SECONDS:-900}"
DISCONNECT_GRACE_SECONDS="${NOMAS_DISPATCHER_DISCONNECT_GRACE_SECONDS:-3}"
ALLOW_WAITLIST="${NOMAS_DISPATCHER_ALLOW_WAITLIST:-true}"
FRONTEND_OUT_DIR="${FRONTEND_OUT_DIR:-dist-dispatcher}"
FRONTEND_DIR="${FRONTEND_DIR:-apps/frontend/${FRONTEND_OUT_DIR}}"
ATTEMPT_DIRS="${NOMAS_USER_ATTEMPT_DIRS:-${NOMAS_USER_ATTEMPT_DIR:-}}"
GALLERY_DIR="${NOMAS_DISPATCHER_GALLERY_DIR:-${NOMAS_GALLERY_SOURCE_DIR:-}}"
NPM_BIN="${NPM_BIN:-$(command -v npm || true)}"

if [[ "${NOMAS_SKIP_DISPATCHER_FRONTEND_BUILD:-false}" =~ ^(1|true|yes|on)$ ]]; then
  if [[ ! -f "${FRONTEND_DIR}/index.html" ]]; then
    echo "Prebuilt dispatcher frontend is missing: ${FRONTEND_DIR}/index.html" >&2
    exit 1
  fi
  echo "Using prebuilt dispatcher frontend in ${FRONTEND_DIR}"
elif [[ -n "${NPM_BIN}" && -x "${NPM_BIN}" ]]; then
  env PATH="$(dirname "${NPM_BIN}"):${PATH}" VITE_MULTI_HOST=true \
    "${NPM_BIN}" --prefix apps/frontend run build -- \
    --outDir "${FRONTEND_OUT_DIR}"
elif [[ -f "${FRONTEND_DIR}/index.html" ]]; then
  echo "npm is unavailable; using the prebuilt dispatcher frontend"
else
  echo "npm is unavailable and ${FRONTEND_DIR}/index.html is missing" >&2
  exit 1
fi

WAITLIST_ARGS=()
if [[ "${ALLOW_WAITLIST,,}" =~ ^(1|true|yes|on)$ ]]; then
  WAITLIST_ARGS+=(--allow-waitlist)
fi

ATTEMPT_ARGS=()
if [[ -n "${ATTEMPT_DIRS}" ]]; then
  while IFS= read -r attempt_dir; do
    [[ -n "${attempt_dir}" ]] && ATTEMPT_ARGS+=(--attempt-dir "${attempt_dir}")
  done < <(printf '%s\n' "${ATTEMPT_DIRS}" | tr ':' '\n')
fi

GALLERY_ARGS=()
if [[ -n "${GALLERY_DIR}" ]]; then
  mkdir -p "${GALLERY_DIR}"
  GALLERY_ARGS+=(--gallery-dir "${GALLERY_DIR}")
fi

exec python apps/seg_ar_dispatcher.py \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --hosts-file "${HOSTS_FILE}" \
  --state-db "${RESOURCE_DB}" \
  --lease-seconds "${LEASE_SECONDS}" \
  --disconnect-grace-seconds "${DISCONNECT_GRACE_SECONDS}" \
  "${WAITLIST_ARGS[@]}" \
  "${ATTEMPT_ARGS[@]}" \
  "${GALLERY_ARGS[@]}" \
  --frontend-dir "${FRONTEND_DIR}"
