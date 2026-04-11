#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${1:-$ROOT_DIR/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE" >&2
  exit 1
fi

if ! command -v sshpass >/dev/null 2>&1; then
  echo "sshpass is required." >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "rsync is required." >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

required_vars=(
  DEPLOY_SSH_HOST
  DEPLOY_SSH_USER
  DEPLOY_SSH_PASSWORD
)

for name in "${required_vars[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required variable: $name" >&2
    exit 1
  fi
done

: "${DEPLOY_SSH_PORT:=22}"
: "${HTE_REMOTE_APP_DIR:=/opt/hormuz-tectonochemical-engine}"
: "${HTE_REMOTE_IMAGE_NAME:=hte-mcp:latest}"
: "${HTE_REMOTE_CONTAINER_NAME:=hte-hormuz-mcp}"
: "${HTE_REMOTE_MCP_PORT:=28766}"
: "${HTE_MCP_TRANSPORT:=streamable-http}"
: "${FASTMCP_HOST:=0.0.0.0}"
: "${FASTMCP_PORT:=8000}"
: "${FASTMCP_STREAMABLE_HTTP_PATH:=/mcp/hormuz}"
: "${HTE_GPU_MODE:=auto}"
: "${HTE_TENSORFLOW_DISTRIBUTION:=auto}"
: "${HTE_DOCKER_BASE_IMAGE:=}"
: "${HTE_ROCM_BASE_IMAGE:=rocm/tensorflow:latest}"
: "${HTE_ROCM_HSA_OVERRIDE_GFX_VERSION:=}"
: "${HTE_REQUIRE_GPU:=false}"
: "${HTE_MCP_STATELESS_HTTP:=true}"
: "${HTE_MCP_MAX_CONCURRENT_REQUESTS:=6}"
: "${HTE_OAUTH_APPROVAL_PASSWORD_HASH:=}"
: "${HTE_OAUTH_PUBLIC_BASE_URL:=}"
: "${HTE_OAUTH_PAGE_TITLE:=Authorize MCP Access}"
: "${HTE_OAUTH_PAGE_SUBTITLE:=Review and approve this MCP client request.}"
: "${HTE_OAUTH_STATE_FILE:=/app/results/state/oauth_state.json}"
: "${HTE_OAUTH_ACCESS_TOKEN_TTL_SECONDS:=3600}"
: "${HTE_OAUTH_REFRESH_TOKEN_TTL_SECONDS:=2592000}"
: "${HTE_OAUTH_AUTHORIZATION_CODE_TTL_SECONDS:=300}"
: "${HTE_ARTIFACT_LINKS_ENABLED:=true}"
: "${HTE_ARTIFACT_INCLUDE_DEFAULT_INPUTS:=true}"
: "${HTE_ARTIFACT_PUBLIC_BASE_URL:=}"
: "${HTE_ARTIFACT_PUBLIC_PATH:=${FASTMCP_STREAMABLE_HTTP_PATH%/}/artifacts}"
: "${HTE_AUDIT_ENABLED:=true}"
: "${HTE_AUDIT_LOG_RESPONSES:=true}"
: "${HTE_AUDIT_MAX_STRING_LENGTH:=20000}"
: "${HTE_EVIDENCE_MAX_ITEMS:=100}"

STACK_LABEL_KEY="com.hte.stack"
STACK_LABEL_VALUE="hormuz-tectonochemical-engine"

SSH_TARGET="${DEPLOY_SSH_USER}@${DEPLOY_SSH_HOST}"
SSH_CMD=(sshpass -p "$DEPLOY_SSH_PASSWORD" ssh -o StrictHostKeyChecking=no -p "$DEPLOY_SSH_PORT" "$SSH_TARGET")
RSYNC_CMD=(sshpass -p "$DEPLOY_SSH_PASSWORD" rsync -az -e "ssh -o StrictHostKeyChecking=no -p $DEPLOY_SSH_PORT")

lower() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]'
}

is_truthy() {
  case "$(lower "$1")" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_https_url() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "$value" ]]; then
    return
  fi
  if [[ "$value" != https://* ]]; then
    echo "$name must start with https:// for ChatGPT-compatible MCP/OAuth endpoints." >&2
    exit 1
  fi
}

resolve_gpu_mode() {
  local requested_mode
  requested_mode="$(lower "$1")"
  case "$requested_mode" in
    none|nvidia|rocm)
      printf '%s' "$requested_mode"
      return
      ;;
    auto)
      if "${SSH_CMD[@]}" "command -v nvidia-smi >/dev/null 2>&1"; then
        printf '%s' "nvidia"
        return
      fi
      if "${SSH_CMD[@]}" "[ -e /dev/kfd ] && ls /dev/dri/renderD* >/dev/null 2>&1"; then
        printf '%s' "rocm"
        return
      fi
      printf '%s' "none"
      return
      ;;
    *)
      echo "HTE_GPU_MODE must be one of: auto, none, nvidia, rocm" >&2
      exit 1
      ;;
  esac
}

resolve_tensorflow_distribution() {
  local requested_distribution
  local gpu_mode
  requested_distribution="$(lower "$1")"
  gpu_mode="$(lower "$2")"
  case "$requested_distribution" in
    auto)
      case "$gpu_mode" in
        rocm) printf '%s' "rocm" ;;
        nvidia) printf '%s' "cuda" ;;
        *) printf '%s' "cpu" ;;
      esac
      ;;
    none)
      printf '%s' "cpu"
      ;;
    cpu|cuda|rocm)
      printf '%s' "$requested_distribution"
      ;;
    *)
      echo "HTE_TENSORFLOW_DISTRIBUTION must be one of: auto, cpu, cuda, rocm" >&2
      exit 1
      ;;
  esac
}

resolve_base_image() {
  local tensorflow_distribution
  tensorflow_distribution="$1"
  if [[ -n "$HTE_DOCKER_BASE_IMAGE" ]]; then
    printf '%s' "$HTE_DOCKER_BASE_IMAGE"
    return
  fi
  if [[ "$tensorflow_distribution" == "rocm" ]]; then
    printf '%s' "$HTE_ROCM_BASE_IMAGE"
    return
  fi
  printf '%s' "python:3.11-slim"
}

detect_gpu_vendor_hint() {
  if "${SSH_CMD[@]}" "lspci -nn 2>/dev/null | grep -Eiq 'nvidia|10de:'"; then
    printf '%s' "nvidia"
    return
  fi
  if "${SSH_CMD[@]}" "lspci -nn 2>/dev/null | grep -Eiq 'advanced micro devices|amd/ati|1002:'"; then
    printf '%s' "amd"
    return
  fi
  if "${SSH_CMD[@]}" "lspci -nn 2>/dev/null | grep -Eiq 'intel|8086:'"; then
    printf '%s' "intel"
    return
  fi
  printf '%s' "unknown"
}

require_https_url HTE_OAUTH_PUBLIC_BASE_URL
require_https_url HTE_ARTIFACT_PUBLIC_BASE_URL

echo "Syncing repository to ${SSH_TARGET}:${HTE_REMOTE_APP_DIR} ..."
"${SSH_CMD[@]}" "mkdir -p '$HTE_REMOTE_APP_DIR'"
"${RSYNC_CMD[@]}" \
  --exclude '.git' \
  --exclude '.env' \
  --exclude '.venv*' \
  --exclude '__pycache__/' \
  --exclude 'results/state/' \
  "$ROOT_DIR/" \
  "${SSH_TARGET}:${HTE_REMOTE_APP_DIR}/"
"${SSH_CMD[@]}" "mkdir -p '$HTE_REMOTE_APP_DIR/results/state'"

RESOLVED_GPU_MODE="$(resolve_gpu_mode "$HTE_GPU_MODE")"
GPU_VENDOR_HINT="$(detect_gpu_vendor_hint)"
RESOLVED_TENSORFLOW_DISTRIBUTION="$(resolve_tensorflow_distribution "$HTE_TENSORFLOW_DISTRIBUTION" "$RESOLVED_GPU_MODE")"
RESOLVED_BASE_IMAGE="$(resolve_base_image "$RESOLVED_TENSORFLOW_DISTRIBUTION")"
INSTALL_TENSORFLOW="true"
if [[ "$RESOLVED_TENSORFLOW_DISTRIBUTION" == "rocm" ]]; then
  INSTALL_TENSORFLOW="false"
fi

if [[ "$RESOLVED_GPU_MODE" == "rocm" && "$RESOLVED_TENSORFLOW_DISTRIBUTION" != "rocm" ]]; then
  echo "Incompatible deployment: resolved GPU mode is 'rocm' but TensorFlow distribution is '${RESOLVED_TENSORFLOW_DISTRIBUTION}'." >&2
  echo "Set HTE_TENSORFLOW_DISTRIBUTION=auto or rocm for AMD/ROCm hosts." >&2
  exit 1
fi
if [[ "$RESOLVED_GPU_MODE" == "nvidia" && "$RESOLVED_TENSORFLOW_DISTRIBUTION" == "rocm" ]]; then
  echo "Incompatible deployment: resolved GPU mode is 'nvidia' but TensorFlow distribution is 'rocm'." >&2
  exit 1
fi

GPU_DOCKER_FLAGS=""
case "$RESOLVED_GPU_MODE" in
  nvidia)
    GPU_DOCKER_FLAGS="--gpus all"
    ;;
  rocm)
    GPU_DOCKER_FLAGS="--device /dev/kfd --device /dev/dri --group-add video"
    ;;
  none)
    GPU_DOCKER_FLAGS=""
    ;;
esac
ROCM_RUNTIME_ENV_FLAGS=""
if [[ "$RESOLVED_GPU_MODE" == "rocm" && -n "${HTE_ROCM_HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  ROCM_RUNTIME_ENV_FLAGS="-e 'HSA_OVERRIDE_GFX_VERSION=${HTE_ROCM_HSA_OVERRIDE_GFX_VERSION}'"
fi
echo "GPU mode request='${HTE_GPU_MODE}', resolved='${RESOLVED_GPU_MODE}'."
echo "Host GPU vendor hint='${GPU_VENDOR_HINT}'."
echo "TensorFlow distribution request='${HTE_TENSORFLOW_DISTRIBUTION}', resolved='${RESOLVED_TENSORFLOW_DISTRIBUTION}'."
echo "Docker base image='${RESOLVED_BASE_IMAGE}'."
if [[ "$RESOLVED_GPU_MODE" == "rocm" ]]; then
  echo "Warning: ROCm mode selected. Ensure host ROCm drivers and a ROCm-compatible TensorFlow build are installed."
fi
if [[ -n "${HTE_ROCM_HSA_OVERRIDE_GFX_VERSION:-}" ]]; then
  echo "Using ROCm HSA override GFX version '${HTE_ROCM_HSA_OVERRIDE_GFX_VERSION}'."
fi

echo "Building Docker image ${HTE_REMOTE_IMAGE_NAME} ..."
"${SSH_CMD[@]}" "cd '$HTE_REMOTE_APP_DIR' && docker build \
  --build-arg 'BASE_IMAGE=${RESOLVED_BASE_IMAGE}' \
  --build-arg 'HTE_TENSORFLOW_DISTRIBUTION=${RESOLVED_TENSORFLOW_DISTRIBUTION}' \
  --build-arg 'INSTALL_TENSORFLOW=${INSTALL_TENSORFLOW}' \
  -t '$HTE_REMOTE_IMAGE_NAME' ."

echo "Preparing container ${HTE_REMOTE_CONTAINER_NAME} ..."
existing_container_id="$("${SSH_CMD[@]}" "docker ps -a --filter 'name=^/${HTE_REMOTE_CONTAINER_NAME}\$' --format '{{.ID}}'")"
if [[ -n "$existing_container_id" ]]; then
  existing_stack_label="$("${SSH_CMD[@]}" "docker inspect -f \"{{ index .Config.Labels \\\"${STACK_LABEL_KEY}\\\" }}\" '$HTE_REMOTE_CONTAINER_NAME'")"
  if [[ "$existing_stack_label" != "$STACK_LABEL_VALUE" ]]; then
    echo "Container name '${HTE_REMOTE_CONTAINER_NAME}' belongs to another stack. Set HTE_REMOTE_CONTAINER_NAME to a different value." >&2
    exit 1
  fi
  "${SSH_CMD[@]}" "docker rm -f '$HTE_REMOTE_CONTAINER_NAME' >/dev/null 2>&1"
fi

echo "Checking TCP port ${HTE_REMOTE_MCP_PORT} ..."
"${SSH_CMD[@]}" "if ss -ltn '( sport = :${HTE_REMOTE_MCP_PORT} )' | grep -q LISTEN; then echo 'Port ${HTE_REMOTE_MCP_PORT} is already in use.' >&2; exit 1; fi"

echo "Starting container ${HTE_REMOTE_CONTAINER_NAME} ..."
"${SSH_CMD[@]}" "docker run -d --name '$HTE_REMOTE_CONTAINER_NAME' --restart unless-stopped \
  --label '${STACK_LABEL_KEY}=${STACK_LABEL_VALUE}' \
  ${GPU_DOCKER_FLAGS} \
  ${ROCM_RUNTIME_ENV_FLAGS} \
  -v '${HTE_REMOTE_APP_DIR}/results:/app/results' \
  -p '${HTE_REMOTE_MCP_PORT}:${FASTMCP_PORT}' \
  -e 'HTE_MCP_TRANSPORT=${HTE_MCP_TRANSPORT}' \
  -e 'HTE_GPU_MODE=${RESOLVED_GPU_MODE}' \
  -e 'HTE_REQUIRE_GPU=${HTE_REQUIRE_GPU}' \
  -e 'HTE_TENSORFLOW_DISTRIBUTION=${RESOLVED_TENSORFLOW_DISTRIBUTION}' \
  -e 'HTE_HOST_GPU_VENDOR_HINT=${GPU_VENDOR_HINT}' \
  -e 'FASTMCP_HOST=${FASTMCP_HOST}' \
  -e 'FASTMCP_PORT=${FASTMCP_PORT}' \
  -e 'FASTMCP_STREAMABLE_HTTP_PATH=${FASTMCP_STREAMABLE_HTTP_PATH}' \
  -e 'HTE_MCP_STATELESS_HTTP=${HTE_MCP_STATELESS_HTTP}' \
  -e 'HTE_MCP_MAX_CONCURRENT_REQUESTS=${HTE_MCP_MAX_CONCURRENT_REQUESTS}' \
  -e 'HTE_OAUTH_APPROVAL_PASSWORD_HASH=${HTE_OAUTH_APPROVAL_PASSWORD_HASH}' \
  -e 'HTE_OAUTH_PUBLIC_BASE_URL=${HTE_OAUTH_PUBLIC_BASE_URL}' \
  -e 'HTE_OAUTH_PAGE_TITLE=${HTE_OAUTH_PAGE_TITLE}' \
  -e 'HTE_OAUTH_PAGE_SUBTITLE=${HTE_OAUTH_PAGE_SUBTITLE}' \
  -e 'HTE_OAUTH_STATE_FILE=${HTE_OAUTH_STATE_FILE}' \
  -e 'HTE_OAUTH_ACCESS_TOKEN_TTL_SECONDS=${HTE_OAUTH_ACCESS_TOKEN_TTL_SECONDS}' \
  -e 'HTE_OAUTH_REFRESH_TOKEN_TTL_SECONDS=${HTE_OAUTH_REFRESH_TOKEN_TTL_SECONDS}' \
  -e 'HTE_OAUTH_AUTHORIZATION_CODE_TTL_SECONDS=${HTE_OAUTH_AUTHORIZATION_CODE_TTL_SECONDS}' \
  -e 'HTE_ARTIFACT_LINKS_ENABLED=${HTE_ARTIFACT_LINKS_ENABLED}' \
  -e 'HTE_ARTIFACT_INCLUDE_DEFAULT_INPUTS=${HTE_ARTIFACT_INCLUDE_DEFAULT_INPUTS}' \
  -e 'HTE_ARTIFACT_PUBLIC_BASE_URL=${HTE_ARTIFACT_PUBLIC_BASE_URL}' \
  -e 'HTE_ARTIFACT_PUBLIC_PATH=${HTE_ARTIFACT_PUBLIC_PATH}' \
  -e 'HTE_AUDIT_ENABLED=${HTE_AUDIT_ENABLED}' \
  -e 'HTE_AUDIT_LOG_RESPONSES=${HTE_AUDIT_LOG_RESPONSES}' \
  -e 'HTE_AUDIT_MAX_STRING_LENGTH=${HTE_AUDIT_MAX_STRING_LENGTH}' \
  -e 'HTE_EVIDENCE_MAX_ITEMS=${HTE_EVIDENCE_MAX_ITEMS}' \
  '$HTE_REMOTE_IMAGE_NAME'"

echo "Allowing firewall port when ufw is active ..."
"${SSH_CMD[@]}" "if command -v ufw >/dev/null 2>&1 && ufw status | grep -q 'Status: active'; then ufw allow '${HTE_REMOTE_MCP_PORT}/tcp'; fi"

echo "Verifying backend runtime resolution ..."
BACKEND_PROBE_JSON="$("${SSH_CMD[@]}" "docker exec '$HTE_REMOTE_CONTAINER_NAME' /bin/sh -lc \"python -c 'import json; from hte.backends import backend_payload; print(json.dumps(backend_payload(preference=\\\"gpu\\\"), separators=(\\\",\\\",\\\":\\\")))'\"")"
echo "Backend probe: ${BACKEND_PROBE_JSON}"
RESOLVED_DEVICE="$(
  python3 -c 'import json,sys; payload=json.loads(sys.argv[1]); print(payload.get("resolved_device",""))' \
    "$BACKEND_PROBE_JSON"
)"
if is_truthy "$HTE_REQUIRE_GPU" && [[ "$RESOLVED_DEVICE" != "/GPU:0" ]]; then
  echo "Deployment failed: HTE_REQUIRE_GPU=true but resolved_device=${RESOLVED_DEVICE}." >&2
  echo "Backend notes:" >&2
  python3 -c 'import json,sys; payload=json.loads(sys.argv[1]); [print(f"- {line}") for line in payload.get("notes", [])]' "$BACKEND_PROBE_JSON" >&2
  exit 1
fi

if [[ -n "$HTE_OAUTH_PUBLIC_BASE_URL" ]]; then
  echo "Service endpoint: ${HTE_OAUTH_PUBLIC_BASE_URL}${FASTMCP_STREAMABLE_HTTP_PATH}"
else
  echo "Service endpoint: http://${DEPLOY_SSH_HOST}:${HTE_REMOTE_MCP_PORT}${FASTMCP_STREAMABLE_HTTP_PATH}"
fi
