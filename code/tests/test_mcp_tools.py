from __future__ import annotations

from hte.backends import backend_payload
from hte.calibration import forecast_horizon, train_forecaster
from hte.config import AppConfig, DataConfig, OptimizationConfig, TrainingConfig
from hte.mcp_tools import alignment_manifest, backend_status, scenario_briefing
from hte.optimization import optimize_control_schedule
from hte.validation import design_validation_protocols


def _test_config() -> AppConfig:
    return AppConfig(
        data=DataConfig(lookback_steps=6),
        training=TrainingConfig(max_epochs=2, patience=1, batch_size=8, lstm_units=(32, 24, 16)),
        optimization=OptimizationConfig(horizon_steps=3, iterations=8),
    )


def test_backend_probe_surface() -> None:
    payload = backend_payload()
    assert "resolved_device" in payload
    assert payload["probe_success"] is True or payload["available"] is False


def test_training_forecast_and_optimization_pipeline() -> None:
    config = _test_config()
    artifacts = train_forecaster(config=config, force_retrain=True)
    forecast = forecast_horizon(steps=3, config=config, force_retrain=False)
    optimization = optimize_control_schedule(config=config, force_retrain=False)

    assert artifacts.metrics["mean_test_mae"] >= 0.0
    assert len(forecast["forecast"]) == 3
    assert len(optimization.optimized_schedule) == 3
    assert optimization.summary["optimized_mean_urea_yield_pct"] >= 0.0


def test_validation_protocols_and_mcp_briefing() -> None:
    config = _test_config()
    protocols = design_validation_protocols(config=config, force_retrain=False)
    manifest = alignment_manifest()
    status = backend_status()
    briefing = scenario_briefing(force_retrain=False)

    assert len(protocols) == 4
    assert manifest["ok"] is True
    assert status["tool"] == "backend_status"
    assert "provenance" in briefing["data"]
