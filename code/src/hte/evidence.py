from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .audit import redact_for_audit
from .paths import EVIDENCE_ROOT, ensure_runtime_directories


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return max(minimum, parsed)


def _optional_text(item: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = item.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence confidence must be a number between 0.0 and 1.0") from exc
    if parsed < 0.0 or parsed > 1.0:
        raise ValueError("evidence confidence must be between 0.0 and 1.0")
    return parsed


def _normalize_evidence_item(item: dict[str, Any], index: int) -> dict[str, Any]:
    source_url = _optional_text(item, "source_url", "url", "link")
    if source_url is None:
        raise ValueError(f"evidence item {index} must include source_url, url, or link")

    normalized: dict[str, Any] = {
        "source_url": source_url,
        "title": _optional_text(item, "title", "headline"),
        "source_name": _optional_text(item, "source_name", "source", "publisher"),
        "published_at": _optional_text(item, "published_at", "date"),
        "retrieved_at": _optional_text(item, "retrieved_at"),
        "excerpt": _optional_text(item, "excerpt", "quote", "snippet"),
        "summary": _optional_text(item, "summary", "effect_summary"),
        "operational_relevance": _optional_text(item, "operational_relevance", "relevance"),
        "confidence": _confidence(item.get("confidence")),
        "mapped_indices": item.get("mapped_indices", item.get("indices", [])),
    }

    extra = {
        str(key): value
        for key, value in item.items()
        if key
        not in {
            "source_url",
            "url",
            "link",
            "title",
            "headline",
            "source_name",
            "source",
            "publisher",
            "published_at",
            "date",
            "retrieved_at",
            "excerpt",
            "quote",
            "snippet",
            "summary",
            "effect_summary",
            "operational_relevance",
            "relevance",
            "confidence",
            "mapped_indices",
            "indices",
        }
    }
    if extra:
        normalized["metadata"] = redact_for_audit(extra)

    return redact_for_audit(normalized)


def record_operational_evidence(
    evidence_items: list[dict[str, Any]],
    analysis_context: str = "",
    inferred_indices: dict[str, Any] | None = None,
    uncertainty_notes: list[str] | None = None,
    *,
    storage_root: Path | None = None,
) -> dict[str, Any]:
    if not evidence_items:
        raise ValueError("at least one evidence item is required")

    max_items = _env_int("HTE_EVIDENCE_MAX_ITEMS", 100)
    if len(evidence_items) > max_items:
        raise ValueError(f"evidence item count exceeds HTE_EVIDENCE_MAX_ITEMS={max_items}")

    normalized_items: list[dict[str, Any]] = []
    for index, item in enumerate(evidence_items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"evidence item {index} must be a JSON object")
        normalized_items.append(_normalize_evidence_item(item, index))

    ensure_runtime_directories()
    root = EVIDENCE_ROOT if storage_root is None else storage_root
    root.mkdir(parents=True, exist_ok=True)

    created_at = datetime.now(timezone.utc)
    record_id = uuid4().hex
    filename = f"{created_at.strftime('%Y%m%dT%H%M%S%fZ')}_{record_id}.json"
    path = root / filename

    payload = {
        "schema": "hte.operational_evidence.v1",
        "record_id": record_id,
        "created_at_utc": created_at.isoformat(),
        "analysis_context": analysis_context,
        "evidence_count": len(normalized_items),
        "evidence_items": normalized_items,
        "inferred_indices": redact_for_audit(inferred_indices or {}),
        "uncertainty_notes": redact_for_audit(uncertainty_notes or []),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "record_id": record_id,
        "evidence_count": len(normalized_items),
        "path": str(path),
        "schema": payload["schema"],
        "required_source_field": "source_url",
        "note": "Evidence was stored as a durable JSON record. Include this path or the published artifact URL in downstream briefings.",
    }
