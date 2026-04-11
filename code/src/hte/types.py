from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TensorFlowProbe:
    available: bool
    version: str | None
    requested_preference: str
    resolved_device: str
    physical_devices: tuple[str, ...]
    visible_devices: tuple[str, ...]
    metal_devices: tuple[str, ...]
    probe_operation: str
    probe_success: bool
    notes: tuple[str, ...]


@dataclass(frozen=True)
class NormalizationStats:
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    feature_mean: np.ndarray
    feature_std: np.ndarray
    target_mean: np.ndarray
    target_std: np.ndarray

    def feature_index(self, column: str) -> int:
        return self.feature_columns.index(column)

    def target_index(self, column: str) -> int:
        return self.target_columns.index(column)

    def feature_scale(self, values: np.ndarray) -> np.ndarray:
        return (values - self.feature_mean) / self.feature_std

    def feature_unscale(self, values: np.ndarray) -> np.ndarray:
        return values * self.feature_std + self.feature_mean

    def target_scale(self, values: np.ndarray) -> np.ndarray:
        return (values - self.target_mean) / self.target_std

    def target_unscale(self, values: np.ndarray) -> np.ndarray:
        return values * self.target_std + self.target_mean

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": list(self.feature_columns),
            "target_columns": list(self.target_columns),
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "target_mean": self.target_mean.tolist(),
            "target_std": self.target_std.tolist(),
        }


@dataclass(frozen=True)
class DatasetBundle:
    dataframe: pd.DataFrame
    timestamps: tuple[str, ...]
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    train_inputs: np.ndarray
    train_targets: np.ndarray
    validation_inputs: np.ndarray
    validation_targets: np.ndarray
    test_inputs: np.ndarray
    test_targets: np.ndarray
    test_timestamps: tuple[str, ...]
    latest_window: np.ndarray
    latest_window_timestamps: tuple[str, ...]
    stats: NormalizationStats
    control_bounds: dict[str, tuple[float, float]]
    training_feature_bounds: dict[str, tuple[float, float]]
    target_delta_std_scaled: np.ndarray


@dataclass(frozen=True)
class TrainingArtifacts:
    backend: TensorFlowProbe
    model_path: Path
    metrics_path: Path
    predictions_path: Path
    history_path: Path
    metrics: dict[str, Any]


@dataclass(frozen=True)
class OptimizationResult:
    baseline_schedule: list[dict[str, float]]
    optimized_schedule: list[dict[str, float]]
    baseline_predictions: list[dict[str, float]]
    optimized_predictions: list[dict[str, float]]
    objective_trace: list[float]
    summary: dict[str, float]
    result_path: Path
    status: str = "ok"
    flags: tuple[str, ...] = ()
    drift: dict[str, Any] | None = None


@dataclass(frozen=True)
class ExperimentProtocol:
    protocol_id: str
    title: str
    objective: str
    setup: dict[str, float | str]
    measurements: dict[str, str]
    expected_outputs: dict[str, float | str]
    acceptance_criteria: dict[str, str]
    source_alignment: tuple[str, ...]
