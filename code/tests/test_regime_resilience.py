from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import hte.calibration as calibration
from hte.config import AppConfig, DataConfig, OptimizationConfig, TrainingConfig
from hte.dataset import build_dataset_bundle_with_scenarios, latest_regime_drift


def _test_config() -> AppConfig:
    return AppConfig(
        data=DataConfig(lookback_steps=6),
        training=TrainingConfig(max_epochs=2, patience=1, batch_size=8, lstm_units=(32, 24, 16)),
        optimization=OptimizationConfig(horizon_steps=3, iterations=8),
    )


def test_scenario_rows_expand_dataset_and_increase_drift() -> None:
    config = _test_config()
    baseline = build_dataset_bundle_with_scenarios(config=config, scenario_rows=None)
    augmented = build_dataset_bundle_with_scenarios(
        config=config,
        scenario_rows=[
            {
                "timestamp": "scenario_regime_shift_1",
                "shipping_risk_index": 10.0,
            }
        ],
    )

    assert len(augmented.dataframe) == len(baseline.dataframe) + 1
    drift = latest_regime_drift(augmented, config.data.exogenous_columns)
    assert drift["out_of_bounds_fraction"] > 0.0
    assert drift["max_abs_zscore"] > 0.0


class _FakePrediction:
    def __init__(self, values: np.ndarray):
        self._values = np.asarray([values], dtype=np.float32)

    def numpy(self) -> np.ndarray:
        return self._values


class _FlakyModel:
    def __init__(self, target_count: int):
        self.target_count = target_count
        self.calls = 0

    def __call__(self, _window, training: bool = False):  # noqa: ANN001, FBT002
        self.calls += 1
        if self.calls == 1:
            return _FakePrediction(np.full((self.target_count,), np.nan, dtype=np.float32))
        return _FakePrediction(np.zeros((self.target_count,), dtype=np.float32))


class _AlwaysNonFiniteModel:
    def __init__(self, target_count: int):
        self.target_count = target_count

    def __call__(self, _window, training: bool = False):  # noqa: ANN001, FBT002
        return _FakePrediction(np.full((self.target_count,), np.nan, dtype=np.float32))


class _ExplosiveModel:
    def __init__(self, target_count: int, magnitude: float):
        self.target_count = target_count
        self.magnitude = magnitude

    def __call__(self, _window, training: bool = False):  # noqa: ANN001, FBT002
        return _FakePrediction(np.full((self.target_count,), self.magnitude, dtype=np.float32))


def test_forecast_uses_persistence_fallback_when_prediction_is_non_finite(monkeypatch) -> None:
    config = _test_config()
    bundle = build_dataset_bundle_with_scenarios(config=config, scenario_rows=None)
    artifacts = SimpleNamespace(backend=SimpleNamespace(resolved_device="/GPU:0"))
    flaky_model = _AlwaysNonFiniteModel(target_count=len(bundle.target_columns))

    monkeypatch.setattr(
        calibration,
        "_load_model_and_bundle",
        lambda *args, **kwargs: (config, bundle, artifacts, flaky_model),  # noqa: ARG005
    )

    forecast = calibration.forecast_horizon(steps=2, config=config, force_retrain=False)

    assert forecast["status"] == "degraded_non_finite_predictions"
    assert forecast["fallback_steps"] == [1, 2]
    for row in forecast["forecast"]:
        for column in bundle.target_columns:
            assert np.isfinite(float(row[column]))


def test_forecast_clips_excessive_step_changes(monkeypatch) -> None:
    config = _test_config()
    bundle = build_dataset_bundle_with_scenarios(config=config, scenario_rows=None)
    artifacts = SimpleNamespace(backend=SimpleNamespace(resolved_device="/GPU:0"))
    explosive_model = _ExplosiveModel(target_count=len(bundle.target_columns), magnitude=1.0e6)

    monkeypatch.setattr(
        calibration,
        "_load_model_and_bundle",
        lambda *args, **kwargs: (config, bundle, artifacts, explosive_model),  # noqa: ARG005
    )

    forecast = calibration.forecast_horizon(steps=2, config=config, force_retrain=False)

    assert forecast["clipped_steps"] == [1, 2]
    for row in forecast["forecast"]:
        for column in bundle.target_columns:
            assert np.isfinite(float(row[column]))
