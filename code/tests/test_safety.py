from __future__ import annotations

import pytest

from hte.safety import load_runtime_safety_config


def test_safety_defaults_are_stable() -> None:
    config = load_runtime_safety_config({})
    assert config.max_concurrent_requests == 6


def test_safety_reads_env_override() -> None:
    config = load_runtime_safety_config({"HTE_MCP_MAX_CONCURRENT_REQUESTS": "10"})
    assert config.max_concurrent_requests == 10


def test_safety_rejects_non_positive_limit() -> None:
    with pytest.raises(ValueError, match="HTE_MCP_MAX_CONCURRENT_REQUESTS must be a positive integer"):
        load_runtime_safety_config({"HTE_MCP_MAX_CONCURRENT_REQUESTS": "0"})
