from __future__ import annotations

import json
import platform
import time
from typing import Any
from uuid import uuid4

from .artifact_publisher import get_artifact_publisher
from .audit import record_tool_request, record_tool_result
from .backends import backend_payload
from .calibration import forecast_horizon, train_forecaster, write_project_artifacts
from .evidence import record_operational_evidence as persist_operational_evidence
from .optimization import optimize_control_schedule
from .paths import CODE_ROOT, DATA_ROOT, LOGS_ROOT, PAPER_ROOT, RESULTS_ROOT, ensure_runtime_directories
from .provenance import latest_artifact_manifest, provenance_payload
from .safety import RequestConcurrencyGuard
from .validation import design_validation_protocols

_REQUEST_GUARD = RequestConcurrencyGuard.from_env()
_ARTIFACT_PUBLISHER = get_artifact_publisher()


def _log_event(tool_name: str, payload: dict[str, object]) -> None:
    ensure_runtime_directories()
    log_path = LOGS_ROOT / "mcp_events.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tool": tool_name, **payload}) + "\n")


def _run_logged(tool_name: str, fn, *args, **kwargs) -> dict[str, object]:
    request_id = uuid4().hex
    started = time.perf_counter()
    record_tool_request(tool_name, request_id, args, kwargs)
    if not _REQUEST_GUARD.try_acquire():
        duration_ms = (time.perf_counter() - started) * 1000.0
        payload = {
            "request_id": request_id,
            "status": "error",
            "duration_ms": duration_ms,
            "error_type": "OverloadedError",
            "error_message": "max concurrent MCP requests reached; retry shortly",
        }
        _log_event(tool_name, payload)
        response = {
            "ok": False,
            "tool": tool_name,
            "request_id": request_id,
            "duration_ms": duration_ms,
            "error": {"type": "OverloadedError", "message": payload["error_message"]},
        }
        record_tool_result(
            tool_name,
            request_id,
            status="error",
            duration_ms=duration_ms,
            error={"type": "OverloadedError", "message": payload["error_message"]},
        )
        return response

    try:
        data = fn(*args, **kwargs)
        duration_ms = (time.perf_counter() - started) * 1000.0
        payload = {"request_id": request_id, "status": "ok", "duration_ms": duration_ms}
        _log_event(tool_name, payload)
        response = {"ok": True, "tool": tool_name, "request_id": request_id, "duration_ms": duration_ms, "data": data}
        try:
            published = _ARTIFACT_PUBLISHER.publish(
                tool_name=tool_name,
                request_id=request_id,
                args=args,
                kwargs=kwargs,
                response_data=data,
            )
        except Exception as publish_exc:
            published = {
                "enabled": False,
                "error": {
                    "type": publish_exc.__class__.__name__,
                    "message": str(publish_exc),
                },
            }
        if published is not None:
            response["artifacts"] = published
        record_tool_result(
            tool_name,
            request_id,
            status="ok",
            duration_ms=duration_ms,
            response_data=data,
        )
        return response
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000.0
        payload = {
            "request_id": request_id,
            "status": "error",
            "duration_ms": duration_ms,
            "error_type": exc.__class__.__name__,
            "error_message": str(exc),
        }
        _log_event(tool_name, payload)
        response = {
            "ok": False,
            "tool": tool_name,
            "request_id": request_id,
            "duration_ms": duration_ms,
            "error": {"type": exc.__class__.__name__, "message": str(exc)},
        }
        record_tool_result(
            tool_name,
            request_id,
            status="error",
            duration_ms=duration_ms,
            error={"type": exc.__class__.__name__, "message": str(exc)},
        )
        return response
    finally:
        _REQUEST_GUARD.release()


def backend_status(preference: str = "gpu") -> dict[str, object]:
    return _run_logged("backend_status", backend_payload, preference)


def alignment_manifest() -> dict[str, object]:
    return _run_logged("alignment_manifest", provenance_payload)


def train_model(
    backend_preference: str = "gpu",
    force_retrain: bool = False,
    scenario_rows: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    return _run_logged(
        "train_model",
        lambda: train_forecaster(
            backend_preference=backend_preference,
            force_retrain=force_retrain,
            scenario_rows=scenario_rows,
        ).metrics,
    )


def forecast_observables(
    steps: int = 6,
    backend_preference: str = "gpu",
    force_retrain: bool = False,
    scenario_rows: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    return _run_logged(
        "forecast_observables",
        forecast_horizon,
        steps,
        None,
        backend_preference,
        force_retrain,
        scenario_rows,
    )


def optimize_schedule(
    backend_preference: str = "gpu",
    force_retrain: bool = False,
    scenario_rows: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    def _payload():
        result = optimize_control_schedule(
            backend_preference=backend_preference,
            force_retrain=force_retrain,
            scenario_rows=scenario_rows,
        )
        return {
            "baseline_schedule": result.baseline_schedule,
            "optimized_schedule": result.optimized_schedule,
            "baseline_predictions": result.baseline_predictions,
            "optimized_predictions": result.optimized_predictions,
            "objective_trace": result.objective_trace,
            "summary": result.summary,
            "result_path": str(result.result_path),
            "status": result.status,
            "flags": list(result.flags),
            "drift": result.drift,
        }

    return _run_logged(
        "optimize_schedule",
        _payload,
    )


def validation_protocols(
    backend_preference: str = "gpu",
    force_retrain: bool = False,
    scenario_rows: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    def _payload():
        protocols = design_validation_protocols(
            backend_preference=backend_preference,
            force_retrain=force_retrain,
            scenario_rows=scenario_rows,
        )
        return [protocol.__dict__ for protocol in protocols]

    return _run_logged("validation_protocols", _payload)


def write_artifacts(
    backend_preference: str = "gpu",
    force_retrain: bool = False,
    scenario_rows: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    return _run_logged(
        "write_artifacts",
        write_project_artifacts,
        None,
        backend_preference,
        force_retrain,
        scenario_rows,
    )


def scenario_briefing(
    backend_preference: str = "gpu",
    force_retrain: bool = False,
    scenario_rows: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    def _payload():
        train_forecaster(
            backend_preference=backend_preference,
            force_retrain=force_retrain,
            scenario_rows=scenario_rows,
        )
        return {
            "backend": backend_payload(preference=backend_preference),
            "provenance": provenance_payload(),
            "artifacts": write_project_artifacts(
                backend_preference=backend_preference,
                force_retrain=force_retrain,
                scenario_rows=scenario_rows,
            ),
            "latest_manifest": latest_artifact_manifest(),
        }

    return _run_logged("scenario_briefing", _payload)


def record_operational_evidence(
    evidence_items: list[dict[str, object]],
    analysis_context: str = "",
    inferred_indices: dict[str, object] | None = None,
    uncertainty_notes: list[str] | None = None,
) -> dict[str, object]:
    return _run_logged(
        "record_operational_evidence",
        persist_operational_evidence,
        evidence_items,
        analysis_context,
        inferred_indices,
        uncertainty_notes,
    )


def host_diagnostics() -> dict[str, object]:
    def _payload():
        return {
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "paths": {
                "code_root": str(CODE_ROOT),
                "data_root": str(DATA_ROOT),
                "results_root": str(RESULTS_ROOT),
                "paper_root": str(PAPER_ROOT),
            },
            "backend": backend_payload(),
            "required_files": {
                "dataset": (DATA_ROOT / "aligned_hormuz_benchmark.csv").exists(),
                "source_manifest": (DATA_ROOT / "source_manifest.json").exists(),
                "paper_tex": (PAPER_ROOT / "paper.tex").exists(),
            },
        }

    return _run_logged("host_diagnostics", _payload)
