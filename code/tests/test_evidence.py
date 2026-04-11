from __future__ import annotations

import json
from pathlib import Path

import pytest

from hte.evidence import record_operational_evidence


def test_record_operational_evidence_requires_source_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source_url"):
        record_operational_evidence([{"title": "missing link"}], storage_root=tmp_path)


def test_record_operational_evidence_persists_normalized_items(tmp_path: Path) -> None:
    result = record_operational_evidence(
        [
            {
                "url": "https://example.com/hormuz-update",
                "headline": "Operator advisory",
                "source": "Example Wire",
                "date": "2026-04-11",
                "effect_summary": "Transit risk remained elevated.",
                "operational_relevance": "shipping_risk_index",
                "confidence": 0.82,
                "access_token": "do-not-store",
            }
        ],
        analysis_context="last 7 days scan",
        inferred_indices={"shipping_risk_index": {"value": 0.7}},
        uncertainty_notes=["single-source item"],
        storage_root=tmp_path,
    )

    path = Path(result["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    item = payload["evidence_items"][0]

    assert payload["schema"] == "hte.operational_evidence.v1"
    assert result["evidence_count"] == 1
    assert item["source_url"] == "https://example.com/hormuz-update"
    assert item["title"] == "Operator advisory"
    assert item["source_name"] == "Example Wire"
    assert item["metadata"]["access_token"] == "[redacted]"
