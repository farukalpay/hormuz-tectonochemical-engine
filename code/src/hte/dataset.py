from __future__ import annotations

from dataclasses import replace

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
    frame = load_aligned_dataset()
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
