from __future__ import annotations

import json
from pathlib import Path

from .paths import DATA_ROOT, RESULTS_ROOT


SOURCE_MANIFEST = DATA_ROOT / "source_manifest.json"


def load_source_manifest() -> dict[str, object]:
    with SOURCE_MANIFEST.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_artifact_manifest() -> dict[str, object]:
    manifest_path = RESULTS_ROOT / "artifact_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def provenance_payload() -> dict[str, object]:
    manifest = load_source_manifest()
    return {
        "profile_id": manifest["profile_id"],
        "generated_on": manifest["generated_on"],
        "sources": manifest["sources"],
        "alignment_map": manifest["alignment_map"],
    }
