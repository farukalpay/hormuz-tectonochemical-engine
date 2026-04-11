from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import forecast_horizon, train_forecaster, write_project_artifacts
from .optimization import optimize_control_schedule
from .validation import design_validation_protocols


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hormuz tectonochemical forecasting and validation workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the LSTM-attention forecaster")
    train_parser.add_argument("--backend", default="gpu")
    train_parser.add_argument("--force-retrain", action="store_true")
    train_parser.add_argument("--scenario-rows-json", default=None)

    forecast_parser = subparsers.add_parser("forecast", help="Forecast the next aligned horizon")
    forecast_parser.add_argument("--steps", type=int, default=6)
    forecast_parser.add_argument("--backend", default="gpu")
    forecast_parser.add_argument("--force-retrain", action="store_true")
    forecast_parser.add_argument("--scenario-rows-json", default=None)

    optimize_parser = subparsers.add_parser("optimize", help="Optimize the control schedule")
    optimize_parser.add_argument("--backend", default="gpu")
    optimize_parser.add_argument("--force-retrain", action="store_true")
    optimize_parser.add_argument("--scenario-rows-json", default=None)

    validate_parser = subparsers.add_parser("validate", help="Emit aligned laboratory protocols")
    validate_parser.add_argument("--backend", default="gpu")
    validate_parser.add_argument("--force-retrain", action="store_true")
    validate_parser.add_argument("--scenario-rows-json", default=None)

    rebuild_parser = subparsers.add_parser("rebuild", help="Rebuild all published artifacts")
    rebuild_parser.add_argument("--backend", default="gpu")
    rebuild_parser.add_argument("--force-retrain", action="store_true")
    rebuild_parser.add_argument("--scenario-rows-json", default=None)
    return parser


def _load_scenario_rows(path: str | None) -> list[dict[str, object]] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("scenario rows JSON must be a list of objects")
    for idx, row in enumerate(payload, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"scenario row {idx} must be a JSON object")
    return payload


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    scenario_rows = _load_scenario_rows(args.scenario_rows_json)

    if args.command == "train":
        payload = train_forecaster(
            backend_preference=args.backend,
            force_retrain=args.force_retrain,
            scenario_rows=scenario_rows,
        ).metrics
    elif args.command == "forecast":
        payload = forecast_horizon(
            steps=args.steps,
            backend_preference=args.backend,
            force_retrain=args.force_retrain,
            scenario_rows=scenario_rows,
        )
    elif args.command == "optimize":
        payload = optimize_control_schedule(
            backend_preference=args.backend,
            force_retrain=args.force_retrain,
            scenario_rows=scenario_rows,
        ).__dict__
    elif args.command == "validate":
        payload = [
            protocol.__dict__
            for protocol in design_validation_protocols(
                backend_preference=args.backend,
                force_retrain=args.force_retrain,
                scenario_rows=scenario_rows,
            )
        ]
    else:
        payload = write_project_artifacts(
            backend_preference=args.backend,
            force_retrain=args.force_retrain,
            scenario_rows=scenario_rows,
        )

    print(json.dumps(payload, indent=2, ensure_ascii=False))
