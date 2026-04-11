from __future__ import annotations

import json
from pathlib import Path

from hte.audit import AuditConfig, REDACTED, record_tool_request, redact_for_audit, write_audit_record


def test_redact_for_audit_removes_secret_values() -> None:
    payload = {
        "authorization": "Bearer abc",
        "nested": {"client_secret": "secret-value", "safe": "visible"},
        "items": [{"password": "pw"}, {"source_url": "https://example.com/report"}],
    }

    redacted = redact_for_audit(payload)

    assert redacted["authorization"] == REDACTED
    assert redacted["nested"]["client_secret"] == REDACTED
    assert redacted["nested"]["safe"] == "visible"
    assert redacted["items"][0]["password"] == REDACTED
    assert redacted["items"][1]["source_url"] == "https://example.com/report"


def test_write_audit_record_creates_json_and_jsonl(tmp_path: Path) -> None:
    config = AuditConfig(enabled=True, root=tmp_path, log_responses=True, max_string_length=100)

    path = write_audit_record(
        "tool_requests",
        "backend_status_abc",
        {"tool": "backend_status", "authorization": "Bearer abc"},
        config=config,
    )

    assert path is not None
    assert path.exists()
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["tool"] == "backend_status"
    assert stored["authorization"] == REDACTED
    assert (tmp_path / "tool_requests.jsonl").exists()


def test_record_tool_request_can_be_disabled(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HTE_AUDIT_ENABLED", "false")
    monkeypatch.setenv("HTE_AUDIT_ROOT", str(tmp_path))

    path = record_tool_request("backend_status", "abc", (), {"preference": "gpu"})

    assert path is None
    assert list(tmp_path.iterdir()) == []
