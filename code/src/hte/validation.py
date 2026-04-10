from __future__ import annotations

import json

from .calibration import _config_tag, forecast_horizon, train_forecaster
from .config import AppConfig, build_app_config
from .paths import RESULTS_ROOT
from .types import ExperimentProtocol


def design_validation_protocols(
    config: AppConfig | None = None,
    backend_preference: str = "gpu",
    force_retrain: bool = False,
) -> list[ExperimentProtocol]:
    app_config = build_app_config() if config is None else config
    tag = _config_tag(app_config)
    training = train_forecaster(config=app_config, backend_preference=backend_preference, force_retrain=force_retrain)
    forecast = forecast_horizon(steps=4, config=app_config, backend_preference=backend_preference, force_retrain=False)
    first_step = forecast["forecast"][0]
    mae = training.metrics["test_mae"]

    protocols = [
        ExperimentProtocol(
            protocol_id="steam-reforming-gc",
            title="Steam reforming and shift closure under constrained feed-gas margin",
            objective="Verify that the aligned forcing envelope reproduces the forecasted H2/CO ratio and methane slip under fixed-bed Ni/Al2O3 operation.",
            setup={
                "catalyst": "Ni/Al2O3 fixed bed, 0.50 g",
                "temperature_c": 780.0,
                "steam_to_carbon_ratio": 2.85,
                "gas_hourly_space_velocity_h-1": 12000.0,
            },
            measurements={
                "gc_tcd_h2_co_ratio": "Use calibrated GC-TCD peak areas for H2 and CO at steady state.",
                "gc_fid_methane_slip_pct": "Use GC-FID methane peak area with external standard calibration.",
                "carbon_balance_pct": "Report inlet vs outlet carbon closure.",
            },
            expected_outputs={
                "expected_h2_co_ratio": float(first_step["h2_co_ratio"]),
                "expected_methane_slip_pct": float(first_step["methane_slip_pct"]),
                "expected_output_type": "Chromatogram peak areas for H2, CO, CH4, CO2 and derived ratio/slip values.",
            },
            acceptance_criteria={
                "h2_co_ratio": f"Accept if measured H2/CO is within +/- {mae['h2_co_ratio']:.3f} of the forecast.",
                "methane_slip_pct": f"Accept if methane slip is within +/- {mae['methane_slip_pct']:.3f} percentage points.",
            },
            source_alignment=("xu_froment_1989", "cen_chemical_sector_2026"),
        ),
        ExperimentProtocol(
            protocol_id="methanol-synthesis-gc-ftir",
            title="CO2 hydrogenation to methanol with GC and ATR-FTIR readout",
            objective="Check whether the forecasted methanol selectivity and methoxy band ratio remain coupled when pressure and recycle are adjusted inside the aligned envelope.",
            setup={
                "catalyst": "Cu/ZnO/Al2O3 fixed bed, 1.00 g",
                "temperature_c": 245.0,
                "pressure_bar": 48.0,
                "feed_ratio_h2_to_co2": 3.0,
            },
            measurements={
                "gc_fid_methanol_selectivity_pct": "Integrate methanol, CO, and CO2 peaks after condensation and calibration.",
                "atr_ftir_methoxy_ratio": "Report the 1030 cm^-1 to 1380 cm^-1 band area ratio.",
                "condensate_mass_balance_pct": "Report condensate yield relative to carbon feed.",
            },
            expected_outputs={
                "expected_methanol_selectivity_pct": float(first_step["methanol_selectivity_pct"]),
                "expected_ftir_methoxy_ratio": float(first_step["ftir_methoxy_ratio"]),
                "expected_output_type": "GC peak areas and ATR-FTIR methoxy-band ratios.",
            },
            acceptance_criteria={
                "methanol_selectivity_pct": f"Accept if selectivity is within +/- {mae['methanol_selectivity_pct']:.3f} percentage points.",
                "ftir_methoxy_ratio": f"Accept if the 1030/1380 ratio is within +/- {mae['ftir_methoxy_ratio']:.3f}.",
            },
            source_alignment=("vanden_bussche_froment_1996", "unctad_hormuz_2026_03_10"),
        ),
        ExperimentProtocol(
            protocol_id="urea-chain-ftir-ic",
            title="Ammonium carbamate to urea closure with FTIR and ion chromatography",
            objective="Test the nitrogen branch directly with an aligned carbamate-to-urea conversion and downstream nitrate assay.",
            setup={
                "reactor": "Stainless steel microautoclave, 50 mL",
                "temperature_c": 185.0,
                "hold_time_min": 120.0,
                "quench_temperature_c": 25.0,
            },
            measurements={
                "isolated_urea_yield_pct": "Dry the isolated solid and report gravimetric yield.",
                "atr_ftir_urea_carbonyl_ratio": "Report the 1680 cm^-1 to 1460 cm^-1 band area ratio.",
                "ic_nitrate_mg_l": "Measure nitrate in the hydrolysis liquor by ion chromatography.",
            },
            expected_outputs={
                "expected_urea_yield_pct": float(first_step["urea_yield_pct"]),
                "expected_ftir_urea_carbonyl_ratio": float(first_step["ftir_urea_carbonyl_ratio"]),
                "expected_nitrate_mg_l": float(first_step["nitrate_mg_l"]),
                "expected_output_type": "Isolated yield %, ATR-FTIR band ratio, ion chromatogram nitrate concentration.",
            },
            acceptance_criteria={
                "urea_yield_pct": f"Accept if isolated yield is within +/- {mae['urea_yield_pct']:.3f} percentage points.",
                "nitrate_mg_l": f"Accept if nitrate is within +/- {mae['nitrate_mg_l']:.3f} mg/L.",
            },
            source_alignment=("iaea_gov_2025_50", "iaea_gov_2026_8"),
        ),
        ExperimentProtocol(
            protocol_id="desalination-conductivity",
            title="Synthetic seawater polishing with conductivity and nitrate validation",
            objective="Verify the water-service branch by coupling chloride load to permeate conductivity and nitrate burden in a bench RO loop.",
            setup={
                "membrane": "Polyamide RO coupon, 200 cm^2",
                "feed_salinity_g_l": 35.0,
                "transmembrane_pressure_bar": 55.0,
                "recovery_ratio": 0.44,
            },
            measurements={
                "permeate_conductivity_uScm": "Measure permeate conductivity after 30 min steady operation.",
                "ic_nitrate_mg_l": "Measure nitrate in the permeate or polishing stream by ion chromatography.",
                "flux_decline_pct": "Report normalized water flux decline over the run.",
            },
            expected_outputs={
                "expected_permeate_conductivity_uScm": float(first_step["permeate_conductivity_uScm"]),
                "expected_nitrate_mg_l": float(first_step["nitrate_mg_l"]),
                "expected_output_type": "Conductivity trace, ion chromatogram nitrate peak, normalized flux decline.",
            },
            acceptance_criteria={
                "permeate_conductivity_uScm": f"Accept if conductivity is within +/- {mae['permeate_conductivity_uScm']:.3f} uS/cm.",
                "nitrate_mg_l": f"Accept if nitrate is within +/- {mae['nitrate_mg_l']:.3f} mg/L.",
            },
            source_alignment=("elimelech_phillip_2011", "unctad_hormuz_2026_03_10"),
        ),
    ]

    with (RESULTS_ROOT / f"validation_protocols_{tag}.json").open("w", encoding="utf-8") as handle:
        json.dump([protocol.__dict__ for protocol in protocols], handle, indent=2)
    return protocols
