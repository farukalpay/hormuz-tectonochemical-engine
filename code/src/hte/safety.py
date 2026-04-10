from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class RuntimeSafetyConfig:
    max_concurrent_requests: int = 6


def _read_positive_int(source: Mapping[str, str], key: str, default: int) -> int:
    raw = source.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{key} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def load_runtime_safety_config(env: Mapping[str, str] | None = None) -> RuntimeSafetyConfig:
    source = os.environ if env is None else env
    return RuntimeSafetyConfig(
        max_concurrent_requests=_read_positive_int(source, "HTE_MCP_MAX_CONCURRENT_REQUESTS", 6),
    )


class RequestConcurrencyGuard:
    def __init__(self, max_concurrent_requests: int) -> None:
        self._semaphore = threading.BoundedSemaphore(max_concurrent_requests)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> RequestConcurrencyGuard:
        config = load_runtime_safety_config(env=env)
        return cls(max_concurrent_requests=config.max_concurrent_requests)

    def try_acquire(self) -> bool:
        return self._semaphore.acquire(blocking=False)

    def release(self) -> None:
        self._semaphore.release()
