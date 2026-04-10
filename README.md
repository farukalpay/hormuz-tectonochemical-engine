# hormuz-tectonochemical-engine

![HORMUZ Engine](img/strait-of-hormuz.png)

[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](#setup)
[![TensorFlow](https://img.shields.io/badge/backend-TensorFlow-orange.svg)](#setup)
[![MCP](https://img.shields.io/badge/interface-MCP--first-green.svg)](#mcp-surface)

Reproducible hydrocarbon-nitrogen-water forecasting for the Strait of Hormuz, with an MCP-first interface and a paper-ready artifact pipeline.

**Built for two jobs:**
- generate a defensible forecasting artifact stack,
- expose one clean MCP call path without forcing readers through the internals first.

## Why this repo exists

Most research-code repos fail both ways: too academic for operators, too operational for paper readers.

This one does not.

You get:
- TensorFlow-first execution with Metal-aware GPU detection and explicit CPU fallback,
- aligned hydrocarbon / nitrogen / water forecasting with LSTM + temporal attention,
- differentiable schedule optimization over a controllable process window,
- reproducible outputs across metrics, forecasts, figures, optimization summaries, and paper artifacts,
- an MCP surface centered on one practical entry point: `scenario_briefing_tool`.

## Setup

```bash id="40x8yq"
git clone https://github.com/farukalpay/hormuz-tectonochemical-engine.git
cd hormuz-tectonochemical-engine
python3 code/scripts/bootstrap_mcp_host.py --venv .venv-tf
.venv-tf/bin/python code/scripts/check_tensorflow_backend.py
````

## Run

Generate data:

```bash id="yj95fa"
.venv-tf/bin/python code/scripts/generate_aligned_dataset.py
```

Rebuild outputs:

```bash id="74n9up"
.venv-tf/bin/python code/scripts/rebuild_outputs.py
```

Start the MCP server:

```bash id="e4k7dn"
.venv-tf/bin/python -m mcp_server.server
```

Direct CLI path:

```bash id="k7gu2d"
.venv-tf/bin/python -m hte.cli rebuild --backend gpu --force-retrain
```

## Docker

Build:

```bash id="h8zubz"
docker build -t hte-mcp .
```

Run MCP server:

```bash id="de8b37"
docker run --rm -i hte-mcp
```

Persist outputs to host:

```bash id="hn58hl"
docker run --rm -it \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/paper:/app/paper" \
  -v "$(pwd)/data:/app/data" \
  hte-mcp python -m hte.cli rebuild --backend cpu --force-retrain
```

## Repository layout

```text
paper/    manuscript source and compiled PDF
code/     package, MCP server, scripts, tests
data/     aligned benchmark data and source manifest
results/  model metrics, forecasts, optimization summaries, figures
```

## MCP surface

Core tools:

* `backend_status_tool`
* `alignment_manifest_tool`
* `train_model_tool`
* `forecast_observables_tool`
* `optimize_schedule_tool`
* `validation_protocols_tool`
* `write_artifacts_tool`
* `scenario_briefing_tool`
* `host_diagnostics_tool`

Start with `scenario_briefing_tool` if you want the shortest path from zero to a usable payload.

## Outputs

Default runs produce artifacts such as:

* `results/model_metrics_lb12_hz1_u128-96-64.json`
* `results/holdout_predictions_lb12_hz1_u128-96-64.csv`
* `results/horizon_forecast_lb12_hz1_u128-96-64.json`
* `results/optimization_summary_lb12_hz1_u128-96-64.json`
* `results/validation_protocols_lb12_hz1_u128-96-64.json`
* `results/figures/holdout_forecast_lb12_hz1_u128-96-64.png`
* `results/figures/optimized_control_schedule_lb12_hz1_u128-96-64.png`
* `paper/paper.pdf`

## Failure modes worth checking first

If a command fails, check these before blaming the model:

* you cloned the repo and `cd`'d into it,
* the virtual environment exists,
* dependencies actually installed,
* TensorFlow backend detection passed,
* Docker runs include volume mounts when outputs need to survive container exit.

Host diagnostics:

```bash id="lejddj"
.venv-tf/bin/python code/scripts/host_doctor.py
```

## Project metadata

* `CITATION.cff`
* `LICENSE`
* `paper/paper.tex`
* `data/source_manifest.json`
