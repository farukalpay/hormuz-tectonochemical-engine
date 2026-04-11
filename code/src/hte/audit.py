from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import AUDIT_ROOT, ensure_runtime_directories

REDACTED = "[redacted]"
_SENSITIVE_KEY_PARTS = (
    "authorization",
    "cookie",
    "password",
    "secret",
    "token",
    "api_key",
    "apikey",
    "client_secret",
    "access_token",
    "refresh_token",
    "code_verifier",
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


@dataclass(frozen=True)
class AuditConfig:
    enabled: bool
    root: Path
    log_responses: bool
    max_string_length: int

    @classmethod
    def from_env(cls) -> "AuditConfig":
        return cls(
            enabled=_env_flag("HTE_AUDIT_ENABLED", True),
            root=Path(os.environ.get("HTE_AUDIT_ROOT", str(AUDIT_ROOT))).expanduser(),
            log_responses=_env_flag("HTE_AUDIT_LOG_RESPONSES", True),
            max_string_length=_env_int("HTE_AUDIT_MAX_STRING_LENGTH", 20000),
        )


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).replace("-", "_").lower()
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def redact_for_audit(value: Any, *, max_string_length: int | None = None) -> Any:
    limit = max_string_length if max_string_length is not None else AuditConfig.from_env().max_string_length
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        omitted = len(value) - limit
        return f"{value[:limit]}... [truncated {omitted} chars]"
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_sensitive_key(key_text):
                redacted[key_text] = REDACTED
            else:
                redacted[key_text] = redact_for_audit(item, max_string_length=limit)
        return redacted
    if isinstance(value, (list, tuple, set)):
        return [redact_for_audit(item, max_string_length=limit) for item in value]
    return redact_for_audit(str(value), max_string_length=limit)


def _now() -> tuple[str, str, str]:
    current = datetime.now(timezone.utc)
    return (
        current.isoformat(),
        current.strftime("%Y%m%d"),
        current.strftime("%Y%m%dT%H%M%S%fZ"),
    )


def _safe_file_part(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value.strip())
    return cleaned.strip("_") or "record"


def write_audit_record(
    category: str,
    name: str,
    payload: dict[str, Any],
    *,
    config: AuditConfig | None = None,
) -> Path | None:
    resolved_config = AuditConfig.from_env() if config is None else config
    if not resolved_config.enabled:
        return None

    ensure_runtime_directories()
    created_at, day, compact = _now()
    record = {
        "created_at_utc": created_at,
        **redact_for_audit(payload, max_string_length=resolved_config.max_string_length),
    }
    category_part = _safe_file_part(category)
    name_part = _safe_file_part(name)
    category_root = resolved_config.root / category_part / day
    category_root.mkdir(parents=True, exist_ok=True)

    record_path = category_root / f"{compact}_{name_part}.json"
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    jsonl_path = resolved_config.root / f"{category_part}.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({**record, "record_path": str(record_path)}, sort_keys=True) + "\n")

    return record_path


def record_tool_request(tool_name: str, request_id: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Path | None:
    return write_audit_record(
        "tool_requests",
        f"{tool_name}_{request_id}",
        {
            "kind": "mcp_tool_request",
            "tool": tool_name,
            "request_id": request_id,
            "args": list(args),
            "kwargs": kwargs,
        },
    )


def record_tool_result(
    tool_name: str,
    request_id: str,
    *,
    status: str,
    duration_ms: float,
    response_data: Any | None = None,
    error: dict[str, Any] | None = None,
) -> Path | None:
    config = AuditConfig.from_env()
    payload: dict[str, Any] = {
        "kind": "mcp_tool_result",
        "tool": tool_name,
        "request_id": request_id,
        "status": status,
        "duration_ms": duration_ms,
    }
    if error is not None:
        payload["error"] = error
    if config.log_responses and response_data is not None:
        payload["response_data"] = response_data
    elif response_data is not None:
        payload["response_logged"] = False
    return write_audit_record("tool_results", f"{tool_name}_{request_id}", payload, config=config)
