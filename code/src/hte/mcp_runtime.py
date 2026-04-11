from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal, Mapping, cast

MCPTransport = Literal["stdio", "sse", "streamable-http"]
VALID_TRANSPORTS = ("stdio", "sse", "streamable-http")


@dataclass(frozen=True)
class MCPRuntimeConfig:
    transport: MCPTransport = "stdio"
    mount_path: str | None = None
    host: str = "127.0.0.1"
    port: int = 8000
    sse_path: str = "/sse"
    message_path: str = "/messages/"
    streamable_http_path: str = "/mcp"
    stateless_http: bool = False


def _read_port(source: Mapping[str, str], key: str, default: int) -> int:
    raw = source.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < 1 or value > 65535:
        raise ValueError(f"{key} must be between 1 and 65535")
    return value


def _read_path(source: Mapping[str, str], key: str, default: str) -> str:
    raw = source.get(key)
    if raw is None or raw.strip() == "":
        return default
    value = raw.strip()
    if not value.startswith("/"):
        raise ValueError(f"{key} must start with '/'")
    return value


def _read_bool(source: Mapping[str, str], key: str, default: bool) -> bool:
    raw = source.get(key)
    if raw is None or raw.strip() == "":
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean")


def load_mcp_runtime_config(env: Mapping[str, str] | None = None) -> MCPRuntimeConfig:
    source = os.environ if env is None else env
    transport = source.get("HTE_MCP_TRANSPORT", "stdio").strip().lower()
    if transport not in VALID_TRANSPORTS:
        allowed = ", ".join(VALID_TRANSPORTS)
        raise ValueError(f"HTE_MCP_TRANSPORT must be one of: {allowed}")
    mount_path = source.get("HTE_MCP_MOUNT_PATH")
    if mount_path is not None:
        mount_path = mount_path.strip() or None
    host = source.get("FASTMCP_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = _read_port(source, "FASTMCP_PORT", 8000)
    sse_path = _read_path(source, "FASTMCP_SSE_PATH", "/sse")
    message_path = _read_path(source, "FASTMCP_MESSAGE_PATH", "/messages/")
    streamable_http_path = _read_path(source, "FASTMCP_STREAMABLE_HTTP_PATH", "/mcp")
    stateless_default = transport == "streamable-http"
    stateless_http = _read_bool(source, "HTE_MCP_STATELESS_HTTP", stateless_default)
    return MCPRuntimeConfig(
        transport=cast(MCPTransport, transport),
        mount_path=mount_path,
        host=host,
        port=port,
        sse_path=sse_path,
        message_path=message_path,
        streamable_http_path=streamable_http_path,
        stateless_http=stateless_http,
    )
