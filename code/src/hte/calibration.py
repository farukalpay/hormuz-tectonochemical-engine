from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .backends import tensorflow_status
from .config import AppConfig, build_app_config
from .dataset import build_dataset_bundle, denormalize_targets, persistence_driver_schedule
from .model import build_forecaster
from .paths import FIGURES_ROOT, MODELS_ROOT, RESULTS_ROOT, ensure_runtime_directories
from .types import DatasetBundle, TrainingArtifacts


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
) -> TrainingArtifacts:
    app_config = build_app_config() if config is None else config
    ensure_runtime_directories()
    backend = tensorflow_status(preference=backend_preference)
    if not backend.available:
        raise RuntimeError("TensorFlow is unavailable in the active environment")

    bundle = build_dataset_bundle(app_config)
    tag = _config_tag(app_config)
    model_path = MODELS_ROOT / f"lstm_attention_forecaster_{tag}.keras"
    metrics_path = RESULTS_ROOT / f"model_metrics_{tag}.json"
    predictions_path = RESULTS_ROOT / f"holdout_predictions_{tag}.csv"
    history_path = RESULTS_ROOT / f"training_history_{tag}.csv"

    if model_path.exists() and metrics_path.exists() and predictions_path.exists() and not force_retrain:
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        return TrainingArtifacts(
            backend=backend,
            model_path=model_path,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
            history_path=history_path,
            metrics=metrics,
        )

    import tensorflow as tf

    tf.keras.utils.set_random_seed(app_config.training.seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    with tf.device(backend.resolved_device):
        model = build_forecaster(app_config, n_features=len(bundle.feature_columns), n_targets=len(bundle.target_columns))
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=app_config.training.patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(2, app_config.training.patience // 2)),
        ]
        history = model.fit(
            bundle.train_inputs,
            bundle.train_targets,
            validation_data=(bundle.validation_inputs, bundle.validation_targets),
            epochs=app_config.training.max_epochs,
            batch_size=app_config.training.batch_size,
            verbose=0,
            callbacks=callbacks,
        )
        predictions_scaled = model.predict(bundle.test_inputs, verbose=0)
        model.save(model_path)

    metrics = _compute_metrics(bundle, predictions_scaled)
    metrics["backend"] = {
        "resolved_device": backend.resolved_device,
        "version": backend.version,
        "notes": list(backend.notes),
    }
    metrics["training"] = {
        "lookback_steps": app_config.data.lookback_steps,
        "max_epochs": app_config.training.max_epochs,
        "batch_size": app_config.training.batch_size,
        "epochs_completed": len(history.history.get("loss", [])),
    }
    metrics["normalization"] = bundle.stats.to_dict()

    _save_predictions(bundle, predictions_scaled, predictions_path)
    _write_history(history_path, history.history)
    metrics["figures"] = _render_training_figures(history.history, predictions_path, tag)

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
):
    app_config = build_app_config() if config is None else config
    artifacts = train_forecaster(config=app_config, backend_preference=backend_preference, force_retrain=force_retrain)
    bundle = build_dataset_bundle(app_config)
    import tensorflow as tf

    model = tf.keras.models.load_model(artifacts.model_path)
    return app_config, bundle, artifacts, model


def forecast_horizon(
    steps: int = 6,
    config: AppConfig | None = None,
    backend_preference: str = "gpu",
    force_retrain: bool = False,
) -> dict[str, object]:
    app_config, bundle, artifacts, model = _load_model_and_bundle(config, backend_preference, force_retrain)
    import tensorflow as tf

    window = tf.convert_to_tensor(bundle.latest_window[None, :, :], dtype=tf.float32)
    future_schedule = persistence_driver_schedule(bundle, app_config, steps)
    target_indices = [bundle.feature_columns.index(column) for column in bundle.target_columns]
    rows: list[dict[str, float | str]] = []

    for step in range(steps):
        predicted_scaled = model(window, training=False).numpy()[0]
        predicted_actual = bundle.stats.target_unscale(predicted_scaled)
        next_row = window.numpy()[0, -1, :].copy()
        next_row[:] = window.numpy()[0, -1, :]
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
        "resolved_device": artifacts.backend.resolved_device,
        "forecast": rows,
        "driver_policy": "persistence plus linear slope clipping within observed bounds",
    }
    with forecast_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    payload["path"] = str(forecast_path)
    return payload


def write_project_artifacts(
    config: AppConfig | None = None,
    backend_preference: str = "gpu",
    force_retrain: bool = False,
) -> dict[str, object]:
    from .optimization import optimize_control_schedule
    from .validation import design_validation_protocols

    app_config = build_app_config() if config is None else config
    tag = _config_tag(app_config)
    artifacts = train_forecaster(config=app_config, backend_preference=backend_preference, force_retrain=force_retrain)
    forecast = forecast_horizon(config=app_config, backend_preference=backend_preference, force_retrain=force_retrain)
    optimization = optimize_control_schedule(config=app_config, backend_preference=backend_preference, force_retrain=force_retrain)
    protocols = design_validation_protocols(config=app_config, backend_preference=backend_preference, force_retrain=force_retrain)
    manifest = {
        "metrics_path": str(artifacts.metrics_path),
        "predictions_path": str(artifacts.predictions_path),
        "forecast_path": forecast["path"],
        "optimization_path": str(optimization.result_path),
        "validation_protocol_count": len(protocols),
        "figures": artifacts.metrics.get("figures", {}),
        "config_tag": tag,
    }
    with (RESULTS_ROOT / "artifact_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest
