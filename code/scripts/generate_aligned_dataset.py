from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"


def _gaussian_pulse(index: np.ndarray, center: int, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((index - center) / width) ** 2)


def build_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2018-01-01", "2026-03-01", freq="MS")
    index = np.arange(len(dates), dtype=np.float32)
    seasonal = np.sin(2.0 * np.pi * index / 12.0)
    secondary = np.cos(2.0 * np.pi * index / 6.0)

    gov_2025_center = int(np.where(dates == pd.Timestamp("2025-09-01"))[0][0])
    stress_2024_center = int(np.where(dates == pd.Timestamp("2024-06-01"))[0][0])
    shock_2026_center = int(np.where(dates == pd.Timestamp("2026-03-01"))[0][0])
    gov_2025_pulse = _gaussian_pulse(index, gov_2025_center, 2.2)
    stress_2024_pulse = _gaussian_pulse(index, stress_2024_center, 2.6)
    shock_2026_pulse = _gaussian_pulse(index, shock_2026_center, 1.5)

    shipping_risk_index = np.clip(
        0.18 + 0.05 * seasonal + 0.10 * secondary + 0.14 * stress_2024_pulse + 0.10 * gov_2025_pulse + 0.28 * shock_2026_pulse + rng.normal(0.0, 0.015, len(dates)),
        0.06,
        0.95,
    )
    insurance_spread_index = np.clip(
        0.16 + 0.62 * shipping_risk_index + 0.04 * seasonal + rng.normal(0.0, 0.012, len(dates)),
        0.08,
        1.15,
    )
    grid_stability_index = np.clip(
        0.93 - 0.12 * shipping_risk_index - 0.10 * stress_2024_pulse - 0.18 * shock_2026_pulse + 0.03 * secondary + rng.normal(0.0, 0.012, len(dates)),
        0.38,
        0.99,
    )
    inspection_access_index = np.clip(
        0.95 - 0.06 * stress_2024_pulse - 0.11 * gov_2025_pulse - 0.20 * shock_2026_pulse + 0.02 * secondary + rng.normal(0.0, 0.01, len(dates)),
        0.22,
        0.99,
    )
    feed_gas_index = np.clip(
        0.94 - 0.18 * shipping_risk_index - 0.10 * (1.0 - grid_stability_index) + 0.04 * secondary + rng.normal(0.0, 0.012, len(dates)),
        0.34,
        1.00,
    )
    chloride_load_index = np.clip(
        0.28 + 0.12 * (1.0 - grid_stability_index) + 0.14 * shipping_risk_index + 0.05 * stress_2024_pulse + 0.03 * gov_2025_pulse + 0.04 * shock_2026_pulse + rng.normal(0.0, 0.01, len(dates)),
        0.12,
        0.88,
    )

    steam_to_carbon_ratio = np.clip(
        2.80 + 0.10 * seasonal + 0.12 * stress_2024_pulse - 0.18 * shock_2026_pulse + 0.22 * (feed_gas_index - 0.75) + rng.normal(0.0, 0.03, len(dates)),
        2.35,
        3.15,
    )
    reformer_temperature_c = np.clip(
        788.0 + 9.0 * secondary - 10.0 * stress_2024_pulse - 18.0 * shock_2026_pulse + 12.0 * (feed_gas_index - 0.70) + rng.normal(0.0, 2.5, len(dates)),
        745.0,
        820.0,
    )
    synthesis_pressure_bar = np.clip(
        47.0 + 1.8 * seasonal - 1.8 * stress_2024_pulse - 2.5 * shock_2026_pulse + 2.4 * (inspection_access_index - 0.80) + rng.normal(0.0, 0.7, len(dates)),
        40.0,
        55.0,
    )
    recycle_ratio = np.clip(
        0.78 + 0.025 * secondary - 0.02 * stress_2024_pulse - 0.045 * shock_2026_pulse + 0.06 * (feed_gas_index - 0.72) + rng.normal(0.0, 0.008, len(dates)),
        0.65,
        0.90,
    )
    ro_recovery_ratio = np.clip(
        0.46 - 0.02 * stress_2024_pulse - 0.04 * shock_2026_pulse - 0.10 * chloride_load_index + 0.04 * grid_stability_index + rng.normal(0.0, 0.006, len(dates)),
        0.30,
        0.55,
    )

    hydrogen_state = np.zeros(len(dates), dtype=np.float32)
    oxygenate_state = np.zeros(len(dates), dtype=np.float32)
    nitrogen_state = np.zeros(len(dates), dtype=np.float32)
    water_state = np.zeros(len(dates), dtype=np.float32)

    hydrogen_state[0] = 0.72
    oxygenate_state[0] = 0.64
    nitrogen_state[0] = 0.68
    water_state[0] = 0.71

    for idx in range(1, len(dates)):
        hydrogen_state[idx] = np.clip(
            0.72 * hydrogen_state[idx - 1]
            + 0.18 * feed_gas_index[idx]
            + 0.16 * (steam_to_carbon_ratio[idx] - 2.40)
            + 0.10 * ((reformer_temperature_c[idx] - 760.0) / 40.0)
            - 0.16 * shipping_risk_index[idx]
            - 0.12 * (1.0 - grid_stability_index[idx])
            + rng.normal(0.0, 0.01),
            0.06,
            1.00,
        )
        oxygenate_state[idx] = np.clip(
            0.68 * oxygenate_state[idx - 1]
            + 0.22 * hydrogen_state[idx]
            + 0.12 * ((synthesis_pressure_bar[idx] - 40.0) / 10.0)
            + 0.16 * recycle_ratio[idx]
            - 0.10 * insurance_spread_index[idx]
            - 0.06 * chloride_load_index[idx]
            + rng.normal(0.0, 0.01),
            0.05,
            1.00,
        )
        nitrogen_state[idx] = np.clip(
            0.70 * nitrogen_state[idx - 1]
            + 0.20 * hydrogen_state[idx]
            + 0.16 * ((synthesis_pressure_bar[idx] - 40.0) / 10.0)
            + 0.16 * recycle_ratio[idx]
            - 0.10 * (1.0 - inspection_access_index[idx])
            - 0.10 * shipping_risk_index[idx]
            + rng.normal(0.0, 0.01),
            0.05,
            1.00,
        )
        water_state[idx] = np.clip(
            0.74 * water_state[idx - 1]
            + 0.22 * grid_stability_index[idx]
            + 0.18 * (1.0 - ro_recovery_ratio[idx])
            + 0.05 * inspection_access_index[idx]
            - 0.16 * chloride_load_index[idx]
            - 0.10 * shipping_risk_index[idx]
            + rng.normal(0.0, 0.01),
            0.05,
            1.00,
        )

    frame = pd.DataFrame(
        {
            "timestamp": dates.strftime("%Y-%m-%d"),
            "shipping_risk_index": shipping_risk_index,
            "insurance_spread_index": insurance_spread_index,
            "grid_stability_index": grid_stability_index,
            "inspection_access_index": inspection_access_index,
            "feed_gas_index": feed_gas_index,
            "chloride_load_index": chloride_load_index,
            "steam_to_carbon_ratio": steam_to_carbon_ratio,
            "reformer_temperature_c": reformer_temperature_c,
            "synthesis_pressure_bar": synthesis_pressure_bar,
            "recycle_ratio": recycle_ratio,
            "ro_recovery_ratio": ro_recovery_ratio,
            "h2_co_ratio": np.clip(
                1.08 + 1.22 * hydrogen_state + 0.28 * (steam_to_carbon_ratio - 2.6) + 0.12 * ((reformer_temperature_c - 760.0) / 40.0) - 0.10 * shipping_risk_index + rng.normal(0.0, 0.02, len(dates)),
                0.95,
                3.20,
            ),
            "methane_slip_pct": np.clip(
                14.5 - 8.8 * hydrogen_state - 1.8 * (steam_to_carbon_ratio - 2.6) - 0.06 * (reformer_temperature_c - 760.0) + 2.2 * shipping_risk_index + 1.2 * (1.0 - grid_stability_index) + rng.normal(0.0, 0.08, len(dates)),
                1.5,
                20.0,
            ),
            "methanol_selectivity_pct": np.clip(
                24.0 + 46.0 * oxygenate_state + 8.0 * ((synthesis_pressure_bar - 45.0) / 5.0) + 11.0 * recycle_ratio - 7.0 * chloride_load_index + rng.normal(0.0, 0.25, len(dates)),
                20.0,
                95.0,
            ),
            "ftir_methoxy_ratio": np.clip(
                0.58 + 0.65 * oxygenate_state + 0.10 * ((synthesis_pressure_bar - 45.0) / 5.0) + 0.10 * recycle_ratio - 0.10 * chloride_load_index + rng.normal(0.0, 0.01, len(dates)),
                0.25,
                1.80,
            ),
            "urea_yield_pct": np.clip(
                28.0 + 48.0 * nitrogen_state + 9.0 * ((synthesis_pressure_bar - 45.0) / 5.0) + 8.0 * recycle_ratio - 4.5 * shipping_risk_index - 6.0 * (1.0 - inspection_access_index) + rng.normal(0.0, 0.3, len(dates)),
                20.0,
                95.0,
            ),
            "ftir_urea_carbonyl_ratio": np.clip(
                0.62 + 0.55 * nitrogen_state + 0.10 * ((synthesis_pressure_bar - 45.0) / 5.0) + 0.08 * recycle_ratio - 0.08 * shipping_risk_index + rng.normal(0.0, 0.01, len(dates)),
                0.30,
                1.70,
            ),
            "nitrate_mg_l": np.clip(
                2.0 + 18.0 * (1.0 - nitrogen_state) + 6.0 * chloride_load_index + 2.5 * (1.0 - grid_stability_index) + rng.normal(0.0, 0.1, len(dates)),
                1.0,
                35.0,
            ),
            "permeate_conductivity_uScm": np.clip(
                120.0 + 420.0 * (1.0 - water_state) + 70.0 * chloride_load_index + 35.0 * shipping_risk_index + 140.0 * ro_recovery_ratio + rng.normal(0.0, 2.0, len(dates)),
                80.0,
                850.0,
            ),
        }
    )
    return frame


def main() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    frame = build_dataset()
    destination = DATA_ROOT / "aligned_hormuz_benchmark.csv"
    frame.to_csv(destination, index=False)
    print(json.dumps({"dataset_path": str(destination), "rows": len(frame), "columns": list(frame.columns)}, indent=2))


if __name__ == "__main__":
    main()
