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
: "${HTE_MCP_MAX_CONCURRENT_REQUESTS:=6}"
: "${HTE_OAUTH_APPROVAL_PASSWORD_HASH:=}"
: "${HTE_OAUTH_PUBLIC_BASE_URL:=}"
: "${HTE_OAUTH_PAGE_TITLE:=Authorize MCP Access}"
: "${HTE_OAUTH_PAGE_SUBTITLE:=Review and approve this MCP client request.}"
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

echo "Syncing repository to ${SSH_TARGET}:${HTE_REMOTE_APP_DIR} ..."
"${SSH_CMD[@]}" "mkdir -p '$HTE_REMOTE_APP_DIR'"
"${RSYNC_CMD[@]}" \
  --exclude '.git' \
  --exclude '.env' \
  --exclude '.venv*' \
  --exclude '__pycache__/' \
  "$ROOT_DIR/" \
  "${SSH_TARGET}:${HTE_REMOTE_APP_DIR}/"

echo "Building Docker image ${HTE_REMOTE_IMAGE_NAME} ..."
"${SSH_CMD[@]}" "cd '$HTE_REMOTE_APP_DIR' && docker build -t '$HTE_REMOTE_IMAGE_NAME' ."

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
  -p '${HTE_REMOTE_MCP_PORT}:${FASTMCP_PORT}' \
  -e 'HTE_MCP_TRANSPORT=${HTE_MCP_TRANSPORT}' \
  -e 'FASTMCP_HOST=${FASTMCP_HOST}' \
  -e 'FASTMCP_PORT=${FASTMCP_PORT}' \
  -e 'FASTMCP_STREAMABLE_HTTP_PATH=${FASTMCP_STREAMABLE_HTTP_PATH}' \
  -e 'HTE_MCP_MAX_CONCURRENT_REQUESTS=${HTE_MCP_MAX_CONCURRENT_REQUESTS}' \
  -e 'HTE_OAUTH_APPROVAL_PASSWORD_HASH=${HTE_OAUTH_APPROVAL_PASSWORD_HASH}' \
  -e 'HTE_OAUTH_PUBLIC_BASE_URL=${HTE_OAUTH_PUBLIC_BASE_URL}' \
  -e 'HTE_OAUTH_PAGE_TITLE=${HTE_OAUTH_PAGE_TITLE}' \
  -e 'HTE_OAUTH_PAGE_SUBTITLE=${HTE_OAUTH_PAGE_SUBTITLE}' \
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

echo "Service endpoint: http://${DEPLOY_SSH_HOST}:${HTE_REMOTE_MCP_PORT}${FASTMCP_STREAMABLE_HTTP_PATH}"
