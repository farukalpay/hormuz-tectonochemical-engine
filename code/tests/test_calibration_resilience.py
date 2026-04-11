from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import hte.calibration as calibration
import numpy as np
import pytest
from hte.config import AppConfig
from hte.dataset import build_dataset_bundle_with_scenarios
from hte.model import build_forecaster


def test_load_model_retrains_when_cached_model_is_incompatible(monkeypatch) -> None:
    initial_artifacts = SimpleNamespace(model_path=Path("/tmp/old.keras"))
    rebuilt_artifacts = SimpleNamespace(model_path=Path("/tmp/new.keras"))
    retrain_flags: list[bool] = []
    load_calls: list[Path] = []

    def fake_train_forecaster(config, backend_preference, force_retrain, scenario_rows=None):  # noqa: ANN001
        retrain_flags.append(force_retrain)
        if force_retrain:
            return rebuilt_artifacts
        return initial_artifacts

    def fake_load_model(path: Path):
        load_calls.append(path)
        if len(load_calls) == 1:
            raise TypeError("incompatible model artifact")
        return "model-ok"

    fake_tf = SimpleNamespace(keras=SimpleNamespace(models=SimpleNamespace(load_model=fake_load_model)))
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    monkeypatch.setattr(calibration, "build_app_config", lambda: "cfg")
    monkeypatch.setattr(calibration, "build_dataset_bundle_with_scenarios", lambda config, scenario_rows=None: "bundle")
    monkeypatch.setattr(calibration, "train_forecaster", fake_train_forecaster)

    app_config, bundle, artifacts, model = calibration._load_model_and_bundle(None, "gpu", False)

    assert app_config == "cfg"
    assert bundle == "bundle"
    assert model == "model-ok"
    assert artifacts is rebuilt_artifacts
    assert retrain_flags == [False, True]
    assert load_calls == [initial_artifacts.model_path, rebuilt_artifacts.model_path]


def test_cached_cpu_artifacts_are_not_reused_when_gpu_is_available() -> None:
    assert not calibration._should_reuse_cached_artifacts(
        {
            "backend": {
                "resolved_device": "/CPU:0",
                "runtime_signature": calibration._current_runtime_signature(backend_version="2.x"),
            }
        },
        backend_requested="/GPU:0",
        backend_version="2.x",
        scenario_rows=None,
        force_retrain=False,
    )


def test_cpu_fallback_allowed_defaults_to_true_for_gpu_requests(monkeypatch) -> None:
    monkeypatch.delenv("HTE_REQUIRE_GPU", raising=False)
    assert calibration._cpu_fallback_allowed("gpu") is True


def test_cpu_fallback_is_disabled_when_gpu_is_required(monkeypatch) -> None:
    monkeypatch.setenv("HTE_REQUIRE_GPU", "true")
    assert calibration._cpu_fallback_allowed("gpu") is False
    assert calibration._cpu_fallback_allowed("cpu") is True


def test_predict_scaled_with_device_rejects_non_finite_gpu_when_cpu_fallback_disabled(monkeypatch) -> None:
    class _FakeDeviceContext:
        def __enter__(self):  # noqa: ANN204
            return None

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN204
            return False

    fake_tf = SimpleNamespace(device=lambda _: _FakeDeviceContext())
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    class _FakeTensor:
        def __init__(self, values: np.ndarray) -> None:
            self._values = values

        def numpy(self) -> np.ndarray:
            return self._values

    class _FakeModel:
        def __call__(self, window, training=False):  # noqa: ANN001, FBT002
            return _FakeTensor(np.array([[np.nan, np.nan]], dtype=np.float32))

    with pytest.raises(RuntimeError, match="CPU fallback is disabled"):
        calibration._predict_scaled_with_device(
            _FakeModel(),
            window=np.zeros((1, 2, 2), dtype=np.float32),
            preferred_device="/GPU:0",
            allow_cpu_fallback=False,
        )


def test_predict_scaled_with_device_uses_cpu_fallback_when_allowed(monkeypatch) -> None:
    class _FakeDeviceContext:
        def __enter__(self):  # noqa: ANN204
            return None

        def __exit__(self, exc_type, exc, tb):  # noqa: ANN001, ANN204
            return False

    fake_tf = SimpleNamespace(device=lambda _: _FakeDeviceContext())
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)

    class _FakeTensor:
        def __init__(self, values: np.ndarray) -> None:
            self._values = values

        def numpy(self) -> np.ndarray:
            return self._values

    class _FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, window, training=False):  # noqa: ANN001, FBT002
            self.calls += 1
            if self.calls == 1:
                return _FakeTensor(np.array([[np.nan, np.nan]], dtype=np.float32))
            return _FakeTensor(np.array([[0.1, 0.2]], dtype=np.float32))

    predicted, used_device = calibration._predict_scaled_with_device(
        _FakeModel(),
        window=np.zeros((1, 2, 2), dtype=np.float32),
        preferred_device="/GPU:0",
        allow_cpu_fallback=True,
    )

    assert used_device == "/CPU:0"
    assert np.isfinite(predicted).all()


def test_cache_without_matching_runtime_signature_is_not_reused() -> None:
    assert not calibration._should_reuse_cached_artifacts(
        {"backend": {"resolved_device": "/GPU:0", "runtime_signature": {"platform": "Darwin"}}},
        backend_requested="/GPU:0",
        backend_version="2.x",
        scenario_rows=None,
        force_retrain=False,
    )


def test_determinism_auto_is_disabled_for_rocm_gpu() -> None:
    assert not calibration._should_enable_op_determinism(
        "auto",
        resolved_device="/GPU:0",
        runtime_context={"tensorflow_distribution": "rocm"},
    )


def test_determinism_auto_stays_enabled_for_cpu() -> None:
    assert calibration._should_enable_op_determinism(
        "auto",
        resolved_device="/CPU:0",
        runtime_context={"tensorflow_distribution": "rocm"},
    )


def test_training_attempt_profiles_include_stabilized_retry() -> None:
    config = AppConfig()
    profiles = calibration._training_attempt_profiles(config)

    assert [name for name, _ in profiles] == ["default", "stabilized", "safe_recurrent"]


def test_training_attempt_profiles_prioritize_safe_variant_for_rocm_gpu() -> None:
    config = AppConfig()
    profiles = calibration._training_attempt_profiles(
        config,
        runtime_context={"tensorflow_distribution": "rocm"},
        requested_device="/GPU:0",
    )

    assert [name for name, _ in profiles] == ["safe_recurrent", "default", "stabilized"]


def test_stabilized_profile_uses_conservative_training_parameters() -> None:
    config = AppConfig()
    stabilized = calibration._stabilized_training_profile(config)

    assert stabilized.training.learning_rate < config.training.learning_rate
    assert stabilized.training.optimizer_epsilon > config.training.optimizer_epsilon
    assert stabilized.training.optimizer_clipnorm < config.training.optimizer_clipnorm
    assert stabilized.training.lstm_unroll is False


def test_safe_recurrent_profile_switches_variant_and_eager_execution() -> None:
    config = AppConfig()
    safe_profile = calibration._safe_recurrent_training_profile(config)

    assert safe_profile.training.model_variant == "safe_recurrent"
    assert safe_profile.training.run_eagerly is True
    assert safe_profile.training.dropout == 0.0
    assert safe_profile.training.optimizer_epsilon > config.training.optimizer_epsilon


def test_build_forecaster_disables_jit_compile_by_default() -> None:
    config = AppConfig()
    bundle = build_dataset_bundle_with_scenarios(config=config, scenario_rows=None)
    target_feature_indices = tuple(bundle.feature_columns.index(column) for column in bundle.target_columns)

    model = build_forecaster(
        config,
        n_features=len(bundle.feature_columns),
        n_targets=len(bundle.target_columns),
        target_feature_indices=target_feature_indices,
        target_feature_mean=bundle.stats.feature_mean[list(target_feature_indices)],
        target_feature_std=bundle.stats.feature_std[list(target_feature_indices)],
        target_mean=bundle.stats.target_mean,
        target_std=bundle.stats.target_std,
    )

    assert model.jit_compile is False


def test_build_forecaster_supports_safe_recurrent_variant() -> None:
    config = AppConfig()
    safe_config = replace(config, training=replace(config.training, model_variant="safe_recurrent"))
    bundle = build_dataset_bundle_with_scenarios(config=safe_config, scenario_rows=None)
    target_feature_indices = tuple(bundle.feature_columns.index(column) for column in bundle.target_columns)

    model = build_forecaster(
        safe_config,
        n_features=len(bundle.feature_columns),
        n_targets=len(bundle.target_columns),
        target_feature_indices=target_feature_indices,
        target_feature_mean=bundle.stats.feature_mean[list(target_feature_indices)],
        target_feature_std=bundle.stats.feature_std[list(target_feature_indices)],
        target_mean=bundle.stats.target_mean,
        target_std=bundle.stats.target_std,
    )

    assert model.name == "hte_safe_recurrent_forecaster"
