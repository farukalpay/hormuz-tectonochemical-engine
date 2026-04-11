from __future__ import annotations

import pytest

from hte.mcp_runtime import load_mcp_runtime_config


def test_runtime_defaults_to_stdio() -> None:
    runtime = load_mcp_runtime_config({})
    assert runtime.transport == "stdio"
    assert runtime.mount_path is None
    assert runtime.host == "127.0.0.1"
    assert runtime.port == 8000
    assert runtime.streamable_http_path == "/mcp"
    assert runtime.stateless_http is False


def test_runtime_accepts_streamable_http() -> None:
    runtime = load_mcp_runtime_config(
        {
            "HTE_MCP_TRANSPORT": "streamable-http",
            "FASTMCP_HOST": "0.0.0.0",
            "FASTMCP_PORT": "28766",
            "FASTMCP_STREAMABLE_HTTP_PATH": "/mcp/hormuz",
        }
    )
    assert runtime.transport == "streamable-http"
    assert runtime.host == "0.0.0.0"
    assert runtime.port == 28766
    assert runtime.streamable_http_path == "/mcp/hormuz"
    assert runtime.stateless_http is True


def test_runtime_allows_explicit_session_mode_override() -> None:
    runtime = load_mcp_runtime_config(
        {
            "HTE_MCP_TRANSPORT": "streamable-http",
            "HTE_MCP_STATELESS_HTTP": "false",
        }
    )
    assert runtime.transport == "streamable-http"
    assert runtime.stateless_http is False


def test_runtime_rejects_unknown_transport() -> None:
    with pytest.raises(ValueError, match="HTE_MCP_TRANSPORT must be one of"):
        load_mcp_runtime_config({"HTE_MCP_TRANSPORT": "grpc"})


def test_runtime_rejects_invalid_port() -> None:
    with pytest.raises(ValueError, match="FASTMCP_PORT must be between 1 and 65535"):
        load_mcp_runtime_config({"FASTMCP_PORT": "70000"})


def test_runtime_rejects_invalid_boolean() -> None:
    with pytest.raises(ValueError, match="HTE_MCP_STATELESS_HTTP must be a boolean"):
        load_mcp_runtime_config({"HTE_MCP_STATELESS_HTTP": "sometimes"})
