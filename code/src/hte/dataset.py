from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import AppConfig
from .paths import DATA_ROOT
from .types import DatasetBundle, NormalizationStats


DATA_FILENAME = "aligned_hormuz_benchmark.csv"


def dataset_path() -> str:
    return str(DATA_ROOT / DATA_FILENAME)


def load_aligned_dataset() -> pd.DataFrame:
    frame = pd.read_csv(DATA_ROOT / DATA_FILENAME)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _coerce_numeric_row(
    source_row: pd.Series,
    row: dict[str, Any],
    feature_columns: tuple[str, ...],
    row_index: int,
) -> pd.Series:
    candidate = source_row.copy()
    for key, value in row.items():
        if key == "timestamp":
            continue
        if key not in feature_columns:
            raise ValueError(f"scenario row {row_index} has unsupported feature column: {key}")
        try:
            candidate[key] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scenario row {row_index} column '{key}' must be numeric") from exc

    for column in feature_columns:
        value = candidate[column]
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"scenario row {row_index} column '{column}' must be numeric") from exc
        if not np.isfinite(numeric):
            raise ValueError(f"scenario row {row_index} column '{column}' must be finite")
        candidate[column] = numeric
    return candidate


def _inject_scenario_rows(
    frame: pd.DataFrame,
    config: AppConfig,
    scenario_rows: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    if not scenario_rows:
        return frame

    augmented = frame.copy()
    feature_columns = config.data.sequence_columns
    if not isinstance(scenario_rows, list):
        raise ValueError("scenario_rows must be a list of row objects")

    last_row = augmented.iloc[-1].copy()
    for idx, row in enumerate(scenario_rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"scenario row {idx} must be a JSON object")
        next_row = _coerce_numeric_row(last_row, row, feature_columns, idx)
        timestamp = row.get("timestamp")
        if timestamp is None:
            timestamp = f"scenario_step_{idx}"
        next_row[config.data.timestamp_column] = str(timestamp)
        augmented.loc[len(augmented)] = next_row
        last_row = next_row
    return augmented


def _build_windows(
    values: np.ndarray,
    targets: np.ndarray,
    timestamps: list[str],
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    inputs: list[np.ndarray] = []
    output_targets: list[np.ndarray] = []
    output_timestamps: list[str] = []
    for start in range(0, len(values) - lookback - horizon + 1):
        target_index = start + lookback + horizon - 1
        inputs.append(values[start : start + lookback])
        output_targets.append(targets[target_index])
        output_timestamps.append(timestamps[target_index])
    return (
        np.asarray(inputs, dtype=np.float32),
        np.asarray(output_targets, dtype=np.float32),
        tuple(output_timestamps),
    )


def build_dataset_bundle(config: AppConfig) -> DatasetBundle:
    return build_dataset_bundle_with_scenarios(config=config, scenario_rows=None)


def build_dataset_bundle_with_scenarios(
    config: AppConfig,
    scenario_rows: list[dict[str, Any]] | None = None,
) -> DatasetBundle:
    frame = load_aligned_dataset()
    frame = _inject_scenario_rows(frame, config, scenario_rows)
    feature_columns = config.data.sequence_columns
    target_columns = config.data.observable_columns
    values = frame.loc[:, feature_columns].to_numpy(dtype=np.float32)
    targets = frame.loc[:, target_columns].to_numpy(dtype=np.float32)
    timestamps = frame.loc[:, config.data.timestamp_column].astype(str).tolist()

    raw_inputs, raw_targets, window_timestamps = _build_windows(
        values=values,
        targets=targets,
        timestamps=timestamps,
        lookback=config.data.lookback_steps,
        horizon=config.data.horizon_steps,
    )

    n_samples = raw_inputs.shape[0]
    train_end = max(1, int(n_samples * config.data.train_fraction))
    validation_end = min(
        n_samples - 1,
        train_end + max(1, int(n_samples * config.data.validation_fraction)),
    )

    train_inputs_raw = raw_inputs[:train_end]
    train_targets_raw = raw_targets[:train_end]
    validation_inputs_raw = raw_inputs[train_end:validation_end]
    validation_targets_raw = raw_targets[train_end:validation_end]
    test_inputs_raw = raw_inputs[validation_end:]
    test_targets_raw = raw_targets[validation_end:]

    feature_mean = train_inputs_raw.reshape(-1, raw_inputs.shape[-1]).mean(axis=0)
    feature_std = train_inputs_raw.reshape(-1, raw_inputs.shape[-1]).std(axis=0)
    target_mean = train_targets_raw.mean(axis=0)
    target_std = train_targets_raw.std(axis=0)
    feature_std = np.where(feature_std < 1.0e-6, 1.0, feature_std)
    target_std = np.where(target_std < 1.0e-6, 1.0, target_std)

    stats = NormalizationStats(
        feature_columns=feature_columns,
        target_columns=target_columns,
        feature_mean=feature_mean.astype(np.float32),
        feature_std=feature_std.astype(np.float32),
        target_mean=target_mean.astype(np.float32),
        target_std=target_std.astype(np.float32),
    )

    def _scale_inputs(block: np.ndarray) -> np.ndarray:
        return ((block - stats.feature_mean) / stats.feature_std).astype(np.float32)

    def _scale_targets(block: np.ndarray) -> np.ndarray:
        return ((block - stats.target_mean) / stats.target_std).astype(np.float32)

    control_bounds = {
        column: (
            float(frame[column].min()),
            float(frame[column].max()),
        )
        for column in config.data.control_columns
    }
    train_target_last_index = min(
        len(frame) - 1,
        train_end + config.data.lookback_steps + config.data.horizon_steps - 2,
    )
    training_frame = frame.iloc[: train_target_last_index + 1]
    training_feature_bounds = {
        column: (
            float(training_frame[column].min()),
            float(training_frame[column].max()),
        )
        for column in feature_columns
    }
    training_targets_actual = training_frame.loc[:, target_columns].to_numpy(dtype=np.float32)
    if len(training_targets_actual) >= 2:
        target_delta_actual = np.diff(training_targets_actual, axis=0)
        target_delta_std_actual = target_delta_actual.std(axis=0)
    else:
        target_delta_std_actual = np.zeros((len(target_columns),), dtype=np.float32)
    target_delta_std_scaled = (target_delta_std_actual / stats.target_std).astype(np.float32)

    return DatasetBundle(
        dataframe=frame,
        timestamps=tuple(timestamps),
        feature_columns=feature_columns,
        target_columns=target_columns,
        train_inputs=_scale_inputs(train_inputs_raw),
        train_targets=_scale_targets(train_targets_raw),
        validation_inputs=_scale_inputs(validation_inputs_raw),
        validation_targets=_scale_targets(validation_targets_raw),
        test_inputs=_scale_inputs(test_inputs_raw),
        test_targets=_scale_targets(test_targets_raw),
        test_timestamps=window_timestamps[validation_end:],
        latest_window=_scale_inputs(raw_inputs[-1:])[0],
        latest_window_timestamps=tuple(timestamps[-config.data.lookback_steps :]),
        stats=stats,
        control_bounds=control_bounds,
        training_feature_bounds=training_feature_bounds,
        target_delta_std_scaled=target_delta_std_scaled,
    )


def denormalize_targets(bundle: DatasetBundle, values: np.ndarray) -> np.ndarray:
    return bundle.stats.target_unscale(values)


def persistence_driver_schedule(bundle: DatasetBundle, config: AppConfig, steps: int) -> np.ndarray:
    frame = bundle.dataframe
    feature_columns = bundle.feature_columns
    schedule_columns = config.data.exogenous_columns + config.data.control_columns
    latest = frame.loc[:, schedule_columns].tail(config.data.lookback_steps)
    bounds = {
        column: (float(frame[column].min()), float(frame[column].max()))
        for column in schedule_columns
    }

    if len(latest) >= 2:
        slope = latest.diff().dropna().mean(axis=0).to_numpy(dtype=np.float32)
        baseline = latest.tail(1).to_numpy(dtype=np.float32)[0]
    else:
        slope = np.zeros((len(schedule_columns),), dtype=np.float32)
        baseline = latest.to_numpy(dtype=np.float32)[-1]

    future_rows: list[np.ndarray] = []
    for step in range(steps):
        row = baseline + slope * float(step + 1)
        clipped = []
        for value, column in zip(row.tolist(), schedule_columns, strict=True):
            lower, upper = bounds[column]
            clipped.append(float(np.clip(value, lower, upper)))
        future_rows.append(np.asarray(clipped, dtype=np.float32))

    schedule = np.asarray(future_rows, dtype=np.float32)
    normalized = np.zeros((steps, len(feature_columns)), dtype=np.float32)
    for index, column in enumerate(schedule_columns):
        feature_index = feature_columns.index(column)
        normalized[:, feature_index] = (
            (schedule[:, index] - bundle.stats.feature_mean[feature_index]) / bundle.stats.feature_std[feature_index]
        )
    return normalized


def latest_regime_drift(bundle: DatasetBundle, columns: tuple[str, ...]) -> dict[str, object]:
    if not columns:
        return {
            "columns": [],
            "max_abs_zscore": 0.0,
            "mean_abs_zscore": 0.0,
            "out_of_bounds_fraction": 0.0,
            "details": {},
        }

    latest = bundle.dataframe.iloc[-1]
    details: dict[str, dict[str, float | bool]] = {}
    zscores: list[float] = []
    oob_count = 0
    for column in columns:
        feature_index = bundle.feature_columns.index(column)
        mean = float(bundle.stats.feature_mean[feature_index])
        std = float(bundle.stats.feature_std[feature_index])
        value = float(latest[column])
        if std < 1.0e-6:
            zscore = 0.0
        else:
            zscore = abs((value - mean) / std)
        lower, upper = bundle.training_feature_bounds[column]
        out_of_bounds = value < lower or value > upper
        if out_of_bounds:
            oob_count += 1
        zscores.append(zscore)
        details[column] = {
            "value": value,
            "train_mean": mean,
            "train_std": std,
            "abs_zscore": zscore,
            "train_min": lower,
            "train_max": upper,
            "out_of_bounds": out_of_bounds,
        }

    return {
        "columns": list(columns),
        "max_abs_zscore": float(max(zscores)),
        "mean_abs_zscore": float(np.mean(zscores)),
        "out_of_bounds_fraction": float(oob_count / len(columns)),
        "details": details,
    }
