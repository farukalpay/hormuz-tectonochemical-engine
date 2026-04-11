from __future__ import annotations

import csv
import json
import os
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .backends import accelerator_context
from .config import AppConfig, build_app_config
from .dataset import (
    build_dataset_bundle_with_scenarios,
    denormalize_targets,
    latest_regime_drift,
    persistence_driver_schedule,
)
from .model import build_forecaster
from .paths import FIGURES_ROOT, MODELS_ROOT, RESULTS_ROOT, ensure_runtime_directories
from .types import DatasetBundle, TrainingArtifacts
from .types import TensorFlowProbe


def _config_tag(config: AppConfig) -> str:
    units = "-".join(str(unit) for unit in config.training.lstm_units)
    return f"lb{config.data.lookback_steps}_hz{config.data.horizon_steps}_u{units}"


def _write_history(path: Path, history: dict[str, list[float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss", "mae", "val_loss", "val_mae"])
        epochs = len(history.get("loss", []))
        for epoch in range(epochs):
            writer.writerow(
                [
                    epoch + 1,
                    history.get("loss", [None])[epoch],
                    history.get("mae", [None])[epoch],
                    history.get("val_loss", [None])[epoch],
                    history.get("val_mae", [None])[epoch],
                ]
            )


def _compute_metrics(bundle: DatasetBundle, predictions_scaled: np.ndarray) -> dict[str, Any]:
    predictions = denormalize_targets(bundle, predictions_scaled)
    observed = denormalize_targets(bundle, bundle.test_targets)
    error = predictions - observed
    mae = np.mean(np.abs(error), axis=0)
    rmse = np.sqrt(np.mean(np.square(error), axis=0))
    metrics = {
        "targets": list(bundle.target_columns),
        "test_mae": {column: float(value) for column, value in zip(bundle.target_columns, mae.tolist(), strict=True)},
        "test_rmse": {column: float(value) for column, value in zip(bundle.target_columns, rmse.tolist(), strict=True)},
        "mean_test_mae": float(mae.mean()),
        "mean_test_rmse": float(rmse.mean()),
    }
    return metrics


def _history_has_non_finite(history: dict[str, list[float]]) -> bool:
    for series in history.values():
        for value in series:
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return True
            if not np.isfinite(numeric):
                return True
    return False


def _validate_bundle_finite(bundle: DatasetBundle) -> None:
    blocks = {
        "train_inputs": bundle.train_inputs,
        "train_targets": bundle.train_targets,
        "validation_inputs": bundle.validation_inputs,
        "validation_targets": bundle.validation_targets,
        "test_inputs": bundle.test_inputs,
        "test_targets": bundle.test_targets,
        "latest_window": bundle.latest_window,
    }
    for name, block in blocks.items():
        if not np.isfinite(block).all():
            raise RuntimeError(f"ModelInputError: non-finite values detected in {name}")


def _resolved_backend_device(artifacts: object) -> str:
    metrics = getattr(artifacts, "metrics", None)
    if isinstance(metrics, dict):
        metrics_backend = metrics.get("backend")
    else:
        metrics_backend = None
    if isinstance(metrics_backend, dict):
        device = metrics_backend.get("resolved_device")
        if isinstance(device, str) and device:
            return device
    backend = getattr(artifacts, "backend", None)
    resolved_device = getattr(backend, "resolved_device", None)
    if isinstance(resolved_device, str) and resolved_device:
        return resolved_device
    return "/CPU:0"


def _cached_backend_device(metrics: dict[str, Any]) -> str | None:
    backend = metrics.get("backend")
    if not isinstance(backend, dict):
        return None
    device = backend.get("resolved_device")
    if isinstance(device, str) and device:
        return device
    return None


def _current_runtime_signature(*, backend_version: str | None) -> dict[str, str]:
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "tensorflow_version": backend_version or "",
    }


def _requested_backend_device(preference: str) -> str:
    normalized = preference.strip().lower() if preference else "gpu"
    if normalized in {"cpu", "force-cpu"}:
        return "/CPU:0"
    return "/GPU:0"


def _read_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _cpu_fallback_allowed(backend_preference: str) -> bool:
    requested_device = _requested_backend_device(backend_preference)
    if requested_device == "/CPU:0":
        return True
    return not _read_env_bool("HTE_REQUIRE_GPU", default=False)


def _stabilized_training_profile(config: AppConfig) -> AppConfig:
    training = replace(
        config.training,
        learning_rate=config.training.learning_rate / 4.0,
        optimizer_epsilon=config.training.optimizer_epsilon * 10.0,
        optimizer_clipnorm=config.training.optimizer_clipnorm * 0.5,
        lstm_unroll=False,
    )
    return replace(config, training=training)


def _safe_recurrent_training_profile(config: AppConfig) -> AppConfig:
    stabilized = _stabilized_training_profile(config)
    training = replace(
        stabilized.training,
        learning_rate=stabilized.training.learning_rate * 0.4,
        optimizer_epsilon=stabilized.training.optimizer_epsilon * 100.0,
        optimizer_clipnorm=stabilized.training.optimizer_clipnorm * 0.5,
        dropout=0.0,
        run_eagerly=True,
        lstm_unroll=True,
        model_variant="safe_recurrent",
    )
    return replace(stabilized, training=training)


def _training_attempt_profiles(
    config: AppConfig,
    *,
    runtime_context: dict[str, object] | None = None,
    requested_device: str = "/GPU:0",
) -> tuple[tuple[str, AppConfig], ...]:
    default_profiles = (
        ("default", config),
        ("stabilized", _stabilized_training_profile(config)),
        ("safe_recurrent", _safe_recurrent_training_profile(config)),
    )
    context = runtime_context or {}
    tensorflow_distribution = str(context.get("tensorflow_distribution") or "").strip().lower()
    if requested_device == "/GPU:0" and tensorflow_distribution == "rocm":
        # On ROCm runtimes, unstable graph attempts can poison the process; prefer the numerically safe profile first.
        return (
            ("safe_recurrent", _safe_recurrent_training_profile(config)),
            ("default", config),
            ("stabilized", _stabilized_training_profile(config)),
        )
    return default_profiles


def _training_backend_stub(preference: str, *, backend_version: str | None) -> TensorFlowProbe:
    requested = preference.strip().lower() if preference else "gpu"
    resolved_device = _requested_backend_device(preference)
    notes = ["TensorFlow XLA JIT is disabled for training stability."]
    if resolved_device == "/GPU:0":
        notes.append("Training skips pre-flight GPU inventory; device health is determined by training attempt.")
    return TensorFlowProbe(
        available=True,
        version=backend_version,
        requested_preference=requested,
        resolved_device=resolved_device,
        physical_devices=(),
        visible_devices=(),
        metal_devices=(),
        probe_operation="training-attempt",
        probe_success=False,
        notes=tuple(notes),
    )


def _materialize_selected_device(tf, attempted_device: str) -> str:
    if attempted_device == "/CPU:0":
        return "/CPU:0"
    try:
        if tf.config.list_logical_devices("GPU"):
            return "/GPU:0"
    except Exception:
        pass
    return "/CPU:0"


def _predict_scaled_with_device(
    model,
    window,
    preferred_device: str,
    *,
    allow_cpu_fallback: bool = True,
) -> tuple[np.ndarray, str]:
    import tensorflow as tf

    candidate_devices = [preferred_device]
    if allow_cpu_fallback and preferred_device != "/CPU:0":
        candidate_devices.append("/CPU:0")

    last_error: Exception | None = None
    last_prediction: np.ndarray | None = None
    last_device = candidate_devices[-1]
    for device_name in candidate_devices:
        try:
            with tf.device(device_name):
                predicted_scaled = model(window, training=False).numpy()[0]
            last_prediction = predicted_scaled.astype(np.float32)
            last_device = device_name
            if np.isfinite(last_prediction).all() or device_name == "/CPU:0":
                return last_prediction, device_name
            if not allow_cpu_fallback and device_name != "/CPU:0":
                raise RuntimeError(
                    "InferenceNonFiniteError: GPU prediction produced non-finite values and CPU fallback is disabled (HTE_REQUIRE_GPU=true)."
                )
        except Exception as exc:
            last_error = exc
            if not allow_cpu_fallback and device_name != "/CPU:0":
                raise RuntimeError(
                    f"InferenceDeviceError: GPU inference failed and CPU fallback is disabled (HTE_REQUIRE_GPU=true): {exc}"
                ) from exc
            continue

    if last_prediction is not None and np.isfinite(last_prediction).all():
        return last_prediction, last_device
    if last_error is not None:
        raise RuntimeError(f"InferenceDeviceError: {last_error}") from last_error
    raise RuntimeError("InferenceDeviceError: no device candidates were available")


def _should_reuse_cached_artifacts(
    metrics: dict[str, Any],
    *,
    backend_requested: str,
    backend_version: str | None,
    scenario_rows: list[dict[str, Any]] | None,
    force_retrain: bool,
) -> bool:
    if force_retrain or scenario_rows is not None:
        return False

    backend = metrics.get("backend")
    cached_runtime_signature = backend.get("runtime_signature") if isinstance(backend, dict) else None
    if cached_runtime_signature != _current_runtime_signature(backend_version=backend_version):
        return False

    cached_device = _cached_backend_device(metrics)
    if backend_requested == "/GPU:0" and cached_device != "/GPU:0":
        return False
    return True


def _should_enable_op_determinism(
    determinism_mode: str,
    *,
    resolved_device: str,
    runtime_context: dict[str, object],
) -> bool:
    normalized = determinism_mode.strip().lower()
    if normalized == "enabled":
        return True
    if normalized == "disabled":
        return False
    if normalized != "auto":
        raise ValueError("training.determinism_mode must be one of: auto, enabled, disabled")

    if resolved_device == "/CPU:0":
        return True
    if str(runtime_context.get("tensorflow_distribution") or "").strip().lower() == "rocm":
        return False
    return True


def _baseline_target_scaled_from_window(bundle: DatasetBundle, window_scaled: np.ndarray) -> np.ndarray:
    target_indices = [bundle.feature_columns.index(column) for column in bundle.target_columns]
    latest_target_feature_scaled = window_scaled[-1, target_indices]
    latest_target_actual = (
        latest_target_feature_scaled * bundle.stats.feature_std[target_indices]
        + bundle.stats.feature_mean[target_indices]
    )
    return bundle.stats.target_scale(latest_target_actual)


def _stabilize_prediction(
    bundle: DatasetBundle,
    window_scaled: np.ndarray,
    predicted_scaled: np.ndarray,
    *,
    clip_sigma: float,
) -> tuple[np.ndarray, bool, bool]:
    baseline_target_scaled = _baseline_target_scaled_from_window(bundle, window_scaled)
    if not np.isfinite(predicted_scaled).all():
        return baseline_target_scaled.copy(), True, False

    delta = predicted_scaled - baseline_target_scaled
    clip_limit = clip_sigma * bundle.target_delta_std_scaled
    clipped_delta = np.clip(delta, -clip_limit, clip_limit)
    clipped = not np.allclose(clipped_delta, delta)
    stabilized = baseline_target_scaled + clipped_delta
    return stabilized.astype(np.float32), False, clipped


def _save_predictions(bundle: DatasetBundle, predictions_scaled: np.ndarray, destination: Path) -> None:
    predictions = denormalize_targets(bundle, predictions_scaled)
    observed = denormalize_targets(bundle, bundle.test_targets)
    rows: list[dict[str, float | str]] = []
    for timestamp, truth_row, pred_row in zip(bundle.test_timestamps, observed, predictions, strict=True):
        row: dict[str, float | str] = {"timestamp": timestamp}
        for column, truth_value, pred_value in zip(bundle.target_columns, truth_row.tolist(), pred_row.tolist(), strict=True):
            row[f"{column}_observed"] = float(truth_value)
            row[f"{column}_predicted"] = float(pred_value)
        rows.append(row)
    import pandas as pd

    pd.DataFrame(rows).to_csv(destination, index=False)


def _render_training_figures(history: dict[str, list[float]], prediction_csv: Path, tag: str) -> dict[str, str]:
    history_path = FIGURES_ROOT / f"training_history_{tag}.png"
    holdout_path = FIGURES_ROOT / f"holdout_forecast_{tag}.png"

    plt.figure(figsize=(8.0, 4.5))
    plt.plot(history.get("loss", []), label="train loss", linewidth=2)
    plt.plot(history.get("val_loss", []), label="validation loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(history_path, dpi=200)
    plt.close()

    import pandas as pd

    frame = pd.read_csv(prediction_csv)
    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.0), sharex=False)
    panels = [
        ("h2_co_ratio", axes[0, 0], "H2/CO"),
        ("methanol_selectivity_pct", axes[0, 1], "Methanol selectivity"),
        ("urea_yield_pct", axes[1, 0], "Urea yield"),
        ("permeate_conductivity_uScm", axes[1, 1], "Permeate conductivity"),
    ]
    x_axis = np.arange(len(frame))
    for column, axis, label in panels:
        axis.plot(x_axis, frame[f"{column}_observed"], label="observed", linewidth=2)
        axis.plot(x_axis, frame[f"{column}_predicted"], label="predicted", linewidth=2)
        axis.set_title(label)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="best")
    figure.tight_layout()
    figure.savefig(holdout_path, dpi=200)
    plt.close(figure)

    return {
        "training_history_figure": str(history_path),
        "holdout_forecast_figure": str(holdout_path),
    }


def train_forecaster(
    config: AppConfig | None = None,
    backend_preference: str = "gpu",
    force_retrain: bool = False,
    scenario_rows: list[dict[str, Any]] | None = None,
) -> TrainingArtifacts:
    app_config = build_app_config() if config is None else config
    ensure_runtime_directories()
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is unavailable in the active environment") from exc

    try:
        tf.config.optimizer.set_jit(False)
    except Exception:
        pass
    backend = _training_backend_stub(backend_preference, backend_version=tf.__version__)

    bundle = build_dataset_bundle_with_scenarios(config=app_config, scenario_rows=scenario_rows)
    _validate_bundle_finite(bundle)
    tag = _config_tag(app_config)
    model_path = MODELS_ROOT / f"lstm_attention_forecaster_{tag}.keras"
    metrics_path = RESULTS_ROOT / f"model_metrics_{tag}.json"
    predictions_path = RESULTS_ROOT / f"holdout_predictions_{tag}.csv"
    history_path = RESULTS_ROOT / f"training_history_{tag}.csv"

    if model_path.exists() and metrics_path.exists() and predictions_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if _should_reuse_cached_artifacts(
            metrics,
            backend_requested=backend.resolved_device,
            backend_version=backend.version,
            scenario_rows=scenario_rows,
            force_retrain=force_retrain,
        ):
            return TrainingArtifacts(
                backend=backend,
                model_path=model_path,
                metrics_path=metrics_path,
                predictions_path=predictions_path,
                history_path=history_path,
                metrics=metrics,
            )

    tf.keras.utils.set_random_seed(app_config.training.seed)
    runtime_context = accelerator_context()
    determinism_enabled = _should_enable_op_determinism(
        app_config.training.determinism_mode,
        resolved_device=backend.resolved_device,
        runtime_context=runtime_context,
    )
    if determinism_enabled:
        try:
            tf.config.experimental.enable_op_determinism()
        except Exception:
            determinism_enabled = False

    attempted_devices: list[dict[str, Any]] = []
    allow_cpu_fallback = _cpu_fallback_allowed(backend_preference)
    candidate_devices = [backend.resolved_device]
    if allow_cpu_fallback and backend.resolved_device != "/CPU:0":
        candidate_devices.append("/CPU:0")

    selected_device: str | None = None
    selected_strategy: str | None = None
    selected_profile_config: AppConfig | None = None
    selected_model = None
    selected_history: dict[str, list[float]] | None = None
    selected_predictions: np.ndarray | None = None
    for device_name in candidate_devices:
        for strategy_name, strategy_config in _training_attempt_profiles(
            app_config,
            runtime_context=runtime_context,
            requested_device=device_name,
        ):
            tf.keras.backend.clear_session()
            with tf.device(device_name):
                target_feature_indices = tuple(bundle.feature_columns.index(column) for column in bundle.target_columns)
                model = build_forecaster(
                    strategy_config,
                    n_features=len(bundle.feature_columns),
                    n_targets=len(bundle.target_columns),
                    target_feature_indices=target_feature_indices,
                    target_feature_mean=bundle.stats.feature_mean[list(target_feature_indices)],
                    target_feature_std=bundle.stats.feature_std[list(target_feature_indices)],
                    target_mean=bundle.stats.target_mean,
                    target_std=bundle.stats.target_std,
                )
                callbacks = [
                    tf.keras.callbacks.TerminateOnNaN(),
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=strategy_config.training.patience,
                        restore_best_weights=True,
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=max(2, strategy_config.training.patience // 2),
                    ),
                ]
                history_obj = model.fit(
                    bundle.train_inputs,
                    bundle.train_targets,
                    validation_data=(bundle.validation_inputs, bundle.validation_targets),
                    epochs=strategy_config.training.max_epochs,
                    batch_size=strategy_config.training.batch_size,
                    verbose=0,
                    callbacks=callbacks,
                )
                predictions_scaled = model.predict(bundle.test_inputs, verbose=0)

            history_dict = history_obj.history
            history_non_finite = _history_has_non_finite(history_dict)
            prediction_non_finite = not np.isfinite(predictions_scaled).all()
            attempt = {
                "device": device_name,
                "strategy": strategy_name,
                "model_variant": strategy_config.training.model_variant,
                "learning_rate": float(strategy_config.training.learning_rate),
                "optimizer_epsilon": float(strategy_config.training.optimizer_epsilon),
                "optimizer_clipnorm": float(strategy_config.training.optimizer_clipnorm),
                "run_eagerly": bool(strategy_config.training.run_eagerly),
                "lstm_unroll": bool(strategy_config.training.lstm_unroll),
                "epochs_completed": len(history_dict.get("loss", [])),
                "history_non_finite": history_non_finite,
                "prediction_non_finite": prediction_non_finite,
            }
            attempted_devices.append(attempt)

            if history_non_finite or prediction_non_finite:
                continue

            selected_device = _materialize_selected_device(tf, device_name)
            selected_strategy = strategy_name
            selected_profile_config = strategy_config
            selected_model = model
            selected_history = history_dict
            selected_predictions = predictions_scaled
            break
        if selected_model is not None:
            break

    if (
        selected_model is None
        or selected_history is None
        or selected_predictions is None
        or selected_device is None
        or selected_strategy is None
        or selected_profile_config is None
    ):
        message = "ModelNonConvergentError: training produced non-finite history/predictions on all candidate devices"
        if not allow_cpu_fallback and backend.resolved_device != "/CPU:0":
            message += " CPU fallback is disabled (HTE_REQUIRE_GPU=true)."
        raise RuntimeError(message)

    selected_model.save(model_path)

    metrics = _compute_metrics(bundle, selected_predictions)
    metrics["backend"] = {
        "requested_device": backend.resolved_device,
        "resolved_device": selected_device,
        "training_strategy": selected_strategy,
        "model_variant": selected_profile_config.training.model_variant,
        "version": backend.version,
        "notes": list(backend.notes),
        "runtime_signature": _current_runtime_signature(backend_version=backend.version),
        "cpu_fallback_allowed": allow_cpu_fallback,
    }
    metrics["training"] = {
        "lookback_steps": app_config.data.lookback_steps,
        "model_variant": selected_profile_config.training.model_variant,
        "max_epochs": selected_profile_config.training.max_epochs,
        "batch_size": selected_profile_config.training.batch_size,
        "run_eagerly": bool(selected_profile_config.training.run_eagerly),
        "epochs_completed": len(selected_history.get("loss", [])),
        "determinism_enabled": determinism_enabled,
    }
    metrics["convergence"] = {
        "status": "ok",
        "attempts": attempted_devices,
        "selected_attempt": {
            "device": selected_device,
            "strategy": selected_strategy,
            "model_variant": selected_profile_config.training.model_variant,
        },
        "cpu_fallback_used": selected_device == "/CPU:0" and backend.resolved_device != "/CPU:0",
    }
    metrics["drift"] = latest_regime_drift(bundle, app_config.data.exogenous_columns)
    metrics["normalization"] = bundle.stats.to_dict()

    _save_predictions(bundle, selected_predictions, predictions_path)
    _write_history(history_path, selected_history)
    metrics["figures"] = _render_training_figures(selected_history, predictions_path, tag)

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    return TrainingArtifacts(
        backend=backend,
        model_path=model_path,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        history_path=history_path,
        metrics=metrics,
    )


def _load_model_and_bundle(
    config: AppConfig | None,
    backend_preference: str,
    force_retrain: bool,
    scenario_rows: list[dict[str, Any]] | None = None,
):
    app_config = build_app_config() if config is None else config
    artifacts = train_forecaster(
        config=app_config,
        backend_preference=backend_preference,
        force_retrain=force_retrain,
        scenario_rows=scenario_rows,
    )
    bundle = build_dataset_bundle_with_scenarios(config=app_config, scenario_rows=scenario_rows)
    import tensorflow as tf

    try:
        model = tf.keras.models.load_model(artifacts.model_path)
    except Exception:
        if force_retrain:
            raise
        # Cached model artifacts can become incompatible across Keras/TensorFlow upgrades.
        # Re-training once keeps forecast/optimization endpoints usable without manual cleanup.
        artifacts = train_forecaster(
            config=app_config,
            backend_preference=backend_preference,
            force_retrain=True,
            scenario_rows=scenario_rows,
        )
        model = tf.keras.models.load_model(artifacts.model_path)
    return app_config, bundle, artifacts, model


def forecast_horizon(
    steps: int = 6,
    config: AppConfig | None = None,
    backend_preference: str = "gpu",
    force_retrain: bool = False,
    scenario_rows: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    app_config, bundle, artifacts, model = _load_model_and_bundle(
        config,
        backend_preference,
        force_retrain,
        scenario_rows=scenario_rows,
    )
    import tensorflow as tf

    window = tf.convert_to_tensor(bundle.latest_window[None, :, :], dtype=tf.float32)
    future_schedule = persistence_driver_schedule(bundle, app_config, steps)
    target_indices = [bundle.feature_columns.index(column) for column in bundle.target_columns]
    inference_device = _requested_backend_device(backend_preference)
    allow_cpu_fallback = _cpu_fallback_allowed(backend_preference)
    fallback_steps: list[int] = []
    clipped_steps: list[int] = []
    rows: list[dict[str, float | str]] = []

    for step in range(steps):
        predicted_scaled, inference_device = _predict_scaled_with_device(
            model,
            window,
            inference_device,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        predicted_scaled, used_fallback, used_clip = _stabilize_prediction(
            bundle,
            window.numpy()[0],
            predicted_scaled,
            clip_sigma=app_config.training.forecast_delta_clip_sigma,
        )
        if used_fallback:
            fallback_steps.append(step + 1)
        if used_clip:
            clipped_steps.append(step + 1)
        predicted_actual = bundle.stats.target_unscale(predicted_scaled)
        next_row = future_schedule[step].copy()
        next_row[target_indices] = predicted_scaled
        timestamp = f"forecast_step_{step + 1}"
        rows.append(
            {
                "timestamp": timestamp,
                **{column: float(value) for column, value in zip(bundle.target_columns, predicted_actual.tolist(), strict=True)},
            }
        )
        next_window = np.concatenate([window.numpy()[0, 1:, :], next_row[None, :]], axis=0)
        window = tf.convert_to_tensor(next_window[None, :, :], dtype=tf.float32)

    forecast_path = RESULTS_ROOT / f"horizon_forecast_{_config_tag(app_config)}.json"
    payload = {
        "steps": steps,
        "resolved_device": inference_device,
        "training_device": _resolved_backend_device(artifacts),
        "forecast": rows,
        "driver_policy": "persistence plus linear slope clipping within observed bounds",
        "cpu_fallback_allowed": allow_cpu_fallback,
        "status": "ok" if not fallback_steps else "degraded_non_finite_predictions",
        "fallback_steps": fallback_steps,
        "clipped_steps": clipped_steps,
        "drift": latest_regime_drift(bundle, app_config.data.exogenous_columns),
    }
    with forecast_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["path"] = str(forecast_path)
    return payload


def write_project_artifacts(
    config: AppConfig | None = None,
    backend_preference: str = "gpu",
    force_retrain: bool = False,
    scenario_rows: list[dict[str, Any]] | None = None,
) -> dict[str, object]:
    from .optimization import optimize_control_schedule
    from .validation import design_validation_protocols

    app_config = build_app_config() if config is None else config
    tag = _config_tag(app_config)
    artifacts = train_forecaster(
        config=app_config,
        backend_preference=backend_preference,
        force_retrain=force_retrain,
        scenario_rows=scenario_rows,
    )
    forecast = forecast_horizon(
        config=app_config,
        backend_preference=backend_preference,
        force_retrain=force_retrain,
        scenario_rows=scenario_rows,
    )
    optimization = optimize_control_schedule(
        config=app_config,
        backend_preference=backend_preference,
        force_retrain=force_retrain,
        scenario_rows=scenario_rows,
    )
    protocols = design_validation_protocols(
        config=app_config,
        backend_preference=backend_preference,
        force_retrain=force_retrain,
        scenario_rows=scenario_rows,
    )
    manifest = {
        "metrics_path": str(artifacts.metrics_path),
        "predictions_path": str(artifacts.predictions_path),
        "forecast_path": forecast["path"],
        "optimization_path": str(optimization.result_path),
        "validation_protocol_count": len(protocols),
        "figures": artifacts.metrics.get("figures", {}),
        "config_tag": tag,
        "forecast_status": forecast.get("status", "ok"),
        "optimization_status": optimization.status,
    }
    with (RESULTS_ROOT / "artifact_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
