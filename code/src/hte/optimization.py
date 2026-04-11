from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from .calibration import _config_tag, _load_model_and_bundle, _stabilize_prediction
from .config import AppConfig, build_app_config
from .dataset import latest_regime_drift, persistence_driver_schedule
from .paths import FIGURES_ROOT, RESULTS_ROOT, ensure_runtime_directories
from .types import OptimizationResult


def _append_prediction_row(window: np.ndarray, row: np.ndarray) -> np.ndarray:
    return np.concatenate([window[1:, :], row[None, :]], axis=0)


def optimize_control_schedule(
    config: AppConfig | None = None,
    backend_preference: str = "gpu",
    force_retrain: bool = False,
    scenario_rows: list[dict[str, object]] | None = None,
) -> OptimizationResult:
    ensure_runtime_directories()
    app_config = build_app_config() if config is None else config
    tag = _config_tag(app_config)
    _, bundle, artifacts, model = _load_model_and_bundle(
        app_config,
        backend_preference,
        force_retrain,
        scenario_rows=scenario_rows,
    )
    future_schedule = persistence_driver_schedule(bundle, app_config, app_config.optimization.horizon_steps)
    control_indices = [bundle.feature_columns.index(column) for column in app_config.data.control_columns]
    target_feature_indices = [bundle.feature_columns.index(column) for column in bundle.target_columns]
    drift = latest_regime_drift(bundle, app_config.data.exogenous_columns)
    status = "ok"
    flags: list[str] = []

    import tensorflow as tf

    control_min = tf.constant(
        [bundle.control_bounds[column][0] for column in app_config.data.control_columns],
        dtype=tf.float32,
    )
    control_max = tf.constant(
        [bundle.control_bounds[column][1] for column in app_config.data.control_columns],
        dtype=tf.float32,
    )
    control_mean = tf.constant(
        [bundle.stats.feature_mean[index] for index in control_indices],
        dtype=tf.float32,
    )
    control_std = tf.constant(
        [bundle.stats.feature_std[index] for index in control_indices],
        dtype=tf.float32,
    )
    target_mean = tf.constant(bundle.stats.target_mean, dtype=tf.float32)
    target_std = tf.constant(bundle.stats.target_std, dtype=tf.float32)
    baseline_actual_controls = np.asarray(
        [
            [
                future_schedule[step, control_indices[idx]] * bundle.stats.feature_std[control_indices[idx]]
                + bundle.stats.feature_mean[control_indices[idx]]
                for idx in range(len(control_indices))
            ]
            for step in range(app_config.optimization.horizon_steps)
        ],
        dtype=np.float32,
    )
    baseline_scaled_controls = (baseline_actual_controls - control_min.numpy()) / (control_max.numpy() - control_min.numpy())
    baseline_scaled_controls = np.clip(baseline_scaled_controls, 1.0e-4, 1.0 - 1.0e-4)
    raw_controls = tf.Variable(np.log(baseline_scaled_controls / (1.0 - baseline_scaled_controls)), dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=app_config.optimization.learning_rate)
    latest_window = tf.constant(bundle.latest_window, dtype=tf.float32)
    objective_trace: list[float] = []
    ordered_objectives = app_config.optimization.objective_weights
    target_lookup = {name: index for index, name in enumerate(bundle.target_columns)}

    def rollout_from_controls(control_actual: tf.Tensor):
        window = latest_window
        predicted_actual_steps = []
        for step in range(app_config.optimization.horizon_steps):
            predicted_scaled = model(window[None, :, :], training=False)[0]
            predicted_actual = predicted_scaled * target_std + target_mean
            predicted_actual_steps.append(predicted_actual)
            next_row = tf.constant(future_schedule[step], dtype=tf.float32)
            next_row = tf.tensor_scatter_nd_update(
                next_row,
                indices=[[index] for index in control_indices],
                updates=(control_actual[step] - control_mean) / control_std,
            )
            next_row = tf.tensor_scatter_nd_update(
                next_row,
                indices=[[index] for index in target_feature_indices],
                updates=predicted_scaled,
            )
            window = tf.concat([window[1:, :], next_row[None, :]], axis=0)
        return tf.stack(predicted_actual_steps, axis=0)

    for _ in range(app_config.optimization.iterations):
        with tf.GradientTape() as tape:
            control_scaled = tf.sigmoid(raw_controls)
            control_actual = control_min + control_scaled * (control_max - control_min)
            predicted_actual = rollout_from_controls(control_actual)
            objective = tf.constant(0.0, dtype=tf.float32)
            for name, weight in ordered_objectives.items():
                objective += tf.constant(weight, dtype=tf.float32) * tf.reduce_mean(
                    predicted_actual[:, target_lookup[name]]
                )
            smoothness = tf.reduce_mean(tf.square(control_actual[1:] - control_actual[:-1]))
            baseline_penalty = tf.reduce_mean(tf.square(control_actual - baseline_actual_controls))
            loss = objective + app_config.optimization.smoothness_weight * smoothness + 0.02 * baseline_penalty
        if not np.isfinite(float(loss.numpy())):
            status = "non_convergent_loss"
            flags.append("optimizer_loss_non_finite")
            break
        gradients = tape.gradient(loss, [raw_controls])
        if gradients[0] is None or not np.isfinite(gradients[0].numpy()).all():
            status = "non_convergent_gradients"
            flags.append("optimizer_gradients_non_finite")
            break
        optimizer.apply_gradients(zip(gradients, [raw_controls]))
        objective_trace.append(float(loss.numpy()))

    optimized_actual_controls = (control_min + tf.sigmoid(raw_controls) * (control_max - control_min)).numpy()
    if status != "ok":
        optimized_actual_controls = baseline_actual_controls.copy()

    def _rollout_numpy(actual_controls: np.ndarray):
        window = bundle.latest_window.copy()
        predictions = []
        schedules = []
        fallback_steps: list[int] = []
        clipped_steps: list[int] = []
        for step in range(app_config.optimization.horizon_steps):
            predicted_scaled = model(window[None, :, :], training=False).numpy()[0]
            predicted_scaled, used_fallback, used_clip = _stabilize_prediction(
                bundle,
                window,
                predicted_scaled,
                clip_sigma=app_config.training.forecast_delta_clip_sigma,
            )
            if used_fallback:
                fallback_steps.append(step + 1)
            if used_clip:
                clipped_steps.append(step + 1)
            predicted_actual = bundle.stats.target_unscale(predicted_scaled)
            predictions.append({column: float(value) for column, value in zip(bundle.target_columns, predicted_actual.tolist(), strict=True)})
            feature_row = future_schedule[step].copy()
            for idx, feature_index in enumerate(control_indices):
                actual_value = actual_controls[step, idx]
                feature_row[feature_index] = (actual_value - bundle.stats.feature_mean[feature_index]) / bundle.stats.feature_std[feature_index]
            feature_row[target_feature_indices] = predicted_scaled
            window = _append_prediction_row(window, feature_row)
            schedules.append(
                {
                    column: float(actual_controls[step, index])
                    for index, column in enumerate(app_config.data.control_columns)
                }
            )
        return schedules, predictions, fallback_steps, clipped_steps

    baseline_schedule, baseline_predictions, baseline_fallback_steps, baseline_clipped_steps = _rollout_numpy(baseline_actual_controls)
    optimized_schedule, optimized_predictions, optimized_fallback_steps, optimized_clipped_steps = _rollout_numpy(optimized_actual_controls)
    if baseline_fallback_steps:
        flags.append(f"baseline_rollout_non_finite_steps={baseline_fallback_steps}")
    if baseline_clipped_steps:
        flags.append(f"baseline_rollout_clipped_steps={baseline_clipped_steps}")
    if optimized_fallback_steps:
        flags.append(f"optimized_rollout_non_finite_steps={optimized_fallback_steps}")
        if status == "ok":
            status = "degraded_non_finite_predictions"
    if optimized_clipped_steps:
        flags.append(f"optimized_rollout_clipped_steps={optimized_clipped_steps}")

    summary = {
        "baseline_mean_methane_slip_pct": float(np.mean([row["methane_slip_pct"] for row in baseline_predictions])),
        "optimized_mean_methane_slip_pct": float(np.mean([row["methane_slip_pct"] for row in optimized_predictions])),
        "baseline_mean_urea_yield_pct": float(np.mean([row["urea_yield_pct"] for row in baseline_predictions])),
        "optimized_mean_urea_yield_pct": float(np.mean([row["urea_yield_pct"] for row in optimized_predictions])),
        "baseline_mean_permeate_conductivity_uScm": float(
            np.mean([row["permeate_conductivity_uScm"] for row in baseline_predictions])
        ),
        "optimized_mean_permeate_conductivity_uScm": float(
            np.mean([row["permeate_conductivity_uScm"] for row in optimized_predictions])
        ),
    }

    figure_path = FIGURES_ROOT / f"optimized_control_schedule_{tag}.png"
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex=True)
    panel_names = [
        "steam_to_carbon_ratio",
        "synthesis_pressure_bar",
        "recycle_ratio",
        "ro_recovery_ratio",
    ]
    for axis, column in zip(axes.flatten(), panel_names, strict=True):
        axis.plot([row[column] for row in baseline_schedule], label="baseline", linewidth=2)
        axis.plot([row[column] for row in optimized_schedule], label="optimized", linewidth=2)
        axis.set_title(column)
        axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="best")
    figure.tight_layout()
    figure.savefig(figure_path, dpi=200)
    plt.close(figure)

    result_path = RESULTS_ROOT / f"optimization_summary_{tag}.json"
    payload = {
        "baseline_schedule": baseline_schedule,
        "optimized_schedule": optimized_schedule,
        "baseline_predictions": baseline_predictions,
        "optimized_predictions": optimized_predictions,
        "objective_trace": objective_trace,
        "summary": summary,
        "figure_path": str(figure_path),
        "resolved_device": (
            artifacts.metrics.get("backend", {}).get("resolved_device")
            if isinstance(getattr(artifacts, "metrics", None), dict)
            else None
        )
        or artifacts.backend.resolved_device,
        "status": status,
        "flags": flags,
        "drift": drift,
    }
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return OptimizationResult(
        baseline_schedule=baseline_schedule,
        optimized_schedule=optimized_schedule,
        baseline_predictions=baseline_predictions,
        optimized_predictions=optimized_predictions,
        objective_trace=objective_trace,
        summary=summary,
        result_path=result_path,
        status=status,
        flags=tuple(flags),
        drift=drift,
    )
