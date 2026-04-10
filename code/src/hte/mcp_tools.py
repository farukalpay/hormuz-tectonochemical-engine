from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from uuid import uuid4

from .backends import backend_payload
from .calibration import forecast_horizon, train_forecaster, write_project_artifacts
from .config import build_app_config
from .optimization import optimize_control_schedule
from .paths import CODE_ROOT, DATA_ROOT, LOGS_ROOT, PAPER_ROOT, RESULTS_ROOT, ensure_runtime_directories
from .provenance import latest_artifact_manifest, provenance_payload
from .validation import design_validation_protocols


def _log_event(tool_name: str, payload: dict[str, object]) -> None:
    ensure_runtime_directories()
    log_path = LOGS_ROOT / "mcp_events.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"tool": tool_name, **payload}) + "\n")


def _run_logged(tool_name: str, fn, *args, **kwargs) -> dict[str, object]:
    request_id = uuid4().hex
    started = time.perf_counter()
    try:
        data = fn(*args, **kwargs)
        duration_ms = (time.perf_counter() - started) * 1000.0
        payload = {"request_id": request_id, "status": "ok", "duration_ms": duration_ms}
        _log_event(tool_name, payload)
        return {"ok": True, "tool": tool_name, "request_id": request_id, "duration_ms": duration_ms, "data": data}
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
        return {
            "ok": False,
            "tool": tool_name,
            "request_id": request_id,
            "duration_ms": duration_ms,
            "error": {"type": exc.__class__.__name__, "message": str(exc)},
        }


def backend_status(preference: str = "gpu") -> dict[str, object]:
    return _run_logged("backend_status", backend_payload, preference)


def alignment_manifest() -> dict[str, object]:
    return _run_logged("alignment_manifest", provenance_payload)


def train_model(backend_preference: str = "gpu", force_retrain: bool = False) -> dict[str, object]:
    return _run_logged(
        "train_model",
        lambda: train_forecaster(backend_preference=backend_preference, force_retrain=force_retrain).metrics,
    )


def forecast_observables(steps: int = 6, backend_preference: str = "gpu", force_retrain: bool = False) -> dict[str, object]:
    return _run_logged(
        "forecast_observables",
        forecast_horizon,
        steps,
        None,
        backend_preference,
        force_retrain,
    )


def optimize_schedule(backend_preference: str = "gpu", force_retrain: bool = False) -> dict[str, object]:
    def _payload():
        result = optimize_control_schedule(backend_preference=backend_preference, force_retrain=force_retrain)
        return {
            "baseline_schedule": result.baseline_schedule,
            "optimized_schedule": result.optimized_schedule,
            "baseline_predictions": result.baseline_predictions,
            "optimized_predictions": result.optimized_predictions,
            "objective_trace": result.objective_trace,
            "summary": result.summary,
            "result_path": str(result.result_path),
        }

    return _run_logged(
        "optimize_schedule",
        _payload,
    )


def validation_protocols(backend_preference: str = "gpu", force_retrain: bool = False) -> dict[str, object]:
    def _payload():
        protocols = design_validation_protocols(backend_preference=backend_preference, force_retrain=force_retrain)
        return [protocol.__dict__ for protocol in protocols]

    return _run_logged("validation_protocols", _payload)


def write_artifacts(backend_preference: str = "gpu", force_retrain: bool = False) -> dict[str, object]:
    return _run_logged(
        "write_artifacts",
        write_project_artifacts,
        None,
        backend_preference,
        force_retrain,
    )


def scenario_briefing(backend_preference: str = "gpu", force_retrain: bool = False) -> dict[str, object]:
    def _payload():
        train_forecaster(backend_preference=backend_preference, force_retrain=force_retrain)
        return {
            "backend": backend_payload(preference=backend_preference),
            "provenance": provenance_payload(),
            "artifacts": write_project_artifacts(backend_preference=backend_preference, force_retrain=force_retrain),
            "latest_manifest": latest_artifact_manifest(),
        }

    return _run_logged("scenario_briefing", _payload)


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
