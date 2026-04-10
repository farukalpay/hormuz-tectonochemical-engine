# Calibration Workflow

## Inputs

- Aligned benchmark data: `data/aligned_hormuz_benchmark.csv`
- Dated source manifest: `data/source_manifest.json`
- Forecast package: `code/src/hte/`

## Numerical Steps

1. Generate or refresh the aligned benchmark with `code/scripts/generate_aligned_dataset.py`.
2. Build a 12-step rolling window over 19 sequence variables.
3. Train the three-layer LSTM + temporal-attention forecaster.
4. Evaluate the holdout split and persist MAE/RMSE per observable.
5. Run the bounded six-step control optimizer.
6. Emit validation protocols whose acceptance bands come directly from out-of-sample MAE.

## Default Commands

```bash
.venv-tf/bin/python code/scripts/generate_aligned_dataset.py
.venv-tf/bin/python code/scripts/rebuild_outputs.py
```

## Outputs

- Metrics: `results/model_metrics_lb12_hz1_u128-96-64.json`
- Holdout forecast: `results/holdout_predictions_lb12_hz1_u128-96-64.csv`
- Horizon forecast: `results/horizon_forecast_lb12_hz1_u128-96-64.json`
- Optimization: `results/optimization_summary_lb12_hz1_u128-96-64.json`
- Validation protocols: `results/validation_protocols_lb12_hz1_u128-96-64.json`

## Backend Policy

- TensorFlow is probed before training.
- If Apple Metal is visible and a small GPU matmul succeeds, training runs on `GPU:0`.
- If the Metal probe fails, the runtime logs the error and uses CPU.
- If TensorFlow is unavailable, diagnostics return an install plan rather than switching frameworks silently.
