from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    lookback_steps: int = 12
    horizon_steps: int = 1
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    timestamp_column: str = "timestamp"
    exogenous_columns: tuple[str, ...] = (
        "shipping_risk_index",
        "insurance_spread_index",
        "grid_stability_index",
        "inspection_access_index",
        "feed_gas_index",
        "chloride_load_index",
    )
    control_columns: tuple[str, ...] = (
        "steam_to_carbon_ratio",
        "reformer_temperature_c",
        "synthesis_pressure_bar",
        "recycle_ratio",
        "ro_recovery_ratio",
    )
    observable_columns: tuple[str, ...] = (
        "h2_co_ratio",
        "methane_slip_pct",
        "methanol_selectivity_pct",
        "ftir_methoxy_ratio",
        "urea_yield_pct",
        "ftir_urea_carbonyl_ratio",
        "nitrate_mg_l",
        "permeate_conductivity_uScm",
    )

    @property
    def sequence_columns(self) -> tuple[str, ...]:
        return self.exogenous_columns + self.control_columns + self.observable_columns


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    max_epochs: int = 72
    learning_rate: float = 1.0e-3
    patience: int = 12
    optimizer_clipnorm: float = 1.0
    optimizer_epsilon: float = 1.0e-7
    determinism_mode: str = "auto"
    run_eagerly: bool = False
    jit_compile: bool = False
    steps_per_execution: int = 1
    residual_output_scale: float = 2.5
    forecast_delta_clip_sigma: float = 3.0
    lstm_units: tuple[int, int, int] = (128, 96, 64)
    attention_heads: int = 4
    dropout: float = 0.12
    lstm_use_cudnn: bool = False
    lstm_unroll: bool = True
    model_variant: str = "stacked_attention"


@dataclass(frozen=True)
class OptimizationConfig:
    horizon_steps: int = 6
    iterations: int = 160
    learning_rate: float = 6.0e-2
    smoothness_weight: float = 0.08
    objective_weights: dict[str, float] = field(
        default_factory=lambda: {
            "h2_co_ratio": -0.18,
            "methane_slip_pct": 0.20,
            "methanol_selectivity_pct": -0.18,
            "urea_yield_pct": -0.18,
            "nitrate_mg_l": 0.12,
            "permeate_conductivity_uScm": 0.14,
        }
    )


@dataclass(frozen=True)
class LoggingConfig:
    mcp_log_filename: str = "mcp_events.jsonl"


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def build_app_config() -> AppConfig:
    return AppConfig()
