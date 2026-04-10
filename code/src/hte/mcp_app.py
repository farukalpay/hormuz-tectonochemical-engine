from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from .mcp_tools import (
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
from .provenance import latest_artifact_manifest, provenance_payload


mcp = FastMCP("hormuz-tectonochemical-engine")


@mcp.tool()
def backend_status_tool(preference: str = "gpu") -> dict[str, object]:
    """Inspect TensorFlow, visible devices, Metal probe status, and CPU fallback notes."""
    return backend_status(preference=preference)


@mcp.tool()
def alignment_manifest_tool() -> dict[str, object]:
    """Expose the dated source manifest and the macro-to-chemistry alignment map."""
    return alignment_manifest()


@mcp.tool()
def train_model_tool(backend_preference: str = "gpu", force_retrain: bool = False) -> dict[str, object]:
    """Train the three-layer LSTM plus temporal-attention forecaster."""
    return train_model(backend_preference=backend_preference, force_retrain=force_retrain)


@mcp.tool()
def forecast_observables_tool(
    steps: int = 6,
    backend_preference: str = "gpu",
    force_retrain: bool = False,
) -> dict[str, object]:
    """Forecast aligned chemical observables for the next horizon."""
    return forecast_observables(steps=steps, backend_preference=backend_preference, force_retrain=force_retrain)


@mcp.tool()
def optimize_schedule_tool(backend_preference: str = "gpu", force_retrain: bool = False) -> dict[str, object]:
    """Run the differentiable multi-objective control optimizer over the future horizon."""
    return optimize_schedule(backend_preference=backend_preference, force_retrain=force_retrain)


@mcp.tool()
def validation_protocols_tool(backend_preference: str = "gpu", force_retrain: bool = False) -> dict[str, object]:
    """Return lab protocols with expected GC, FTIR, IC, yield, and conductivity outputs."""
    return validation_protocols(backend_preference=backend_preference, force_retrain=force_retrain)


@mcp.tool()
def write_artifacts_tool(backend_preference: str = "gpu", force_retrain: bool = False) -> dict[str, object]:
    """Rebuild model, forecast, optimization, figures, and manifest artifacts."""
    return write_artifacts(backend_preference=backend_preference, force_retrain=force_retrain)


@mcp.tool()
def scenario_briefing_tool(backend_preference: str = "gpu", force_retrain: bool = False) -> dict[str, object]:
    """Return the shortest single-call briefing payload for MCP clients."""
    return scenario_briefing(backend_preference=backend_preference, force_retrain=force_retrain)


@mcp.tool()
def host_diagnostics_tool() -> dict[str, object]:
    """Inspect host readiness, file layout, and backend availability."""
    return host_diagnostics()


@mcp.resource(
    "hte://alignment/sources",
    name="Aligned Sources",
    description="Dated online sources and chemistry references used by the repository.",
    mime_type="application/json",
)
def aligned_sources_resource() -> str:
    return json.dumps(provenance_payload(), indent=2)


@mcp.resource(
    "hte://results/latest",
    name="Latest Results",
    description="Most recent generated artifact manifest.",
    mime_type="application/json",
)
def latest_results_resource() -> str:
    return json.dumps(latest_artifact_manifest(), indent=2)


@mcp.prompt(
    name="hte_briefing_order",
    title="HTE Briefing Order",
    description="Recommended tool order for new MCP clients.",
)
def briefing_order_prompt() -> str:
    return (
        "Call backend_status_tool, then alignment_manifest_tool, then train_model_tool, "
        "then forecast_observables_tool, optimize_schedule_tool, and validation_protocols_tool."
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
