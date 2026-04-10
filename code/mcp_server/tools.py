from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = str(Path(__file__).resolve().parents[1] / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hte.mcp_tools import (  # noqa: E402
    alignment_manifest,
    backend_status,
    forecast_observables,
    host_diagnostics,
    optimize_schedule,
    scenario_briefing,
    train_model,
    validation_protocols,
    write_artifacts,
)

__all__ = [
    "alignment_manifest",
    "backend_status",
    "forecast_observables",
    "host_diagnostics",
    "optimize_schedule",
    "scenario_briefing",
    "train_model",
    "validation_protocols",
    "write_artifacts",
]
