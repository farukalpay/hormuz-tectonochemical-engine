# hormuz-tectonochemical-engine

![HORMUZ Engine](img/strait-of-hormuz.png)

Reproducible hydrocarbon-nitrogen-water forecasting stack for the Strait of Hormuz problem, with an MCP-first surface and a paper-ready artifact pipeline.

Repository: [github.com/farukalpay/hormuz-tectonochemical-engine](https://github.com/farukalpay/hormuz-tectonochemical-engine)

This repo is built for two readers at once:

- manuscript readers who want the equations, experiments, figures, and PDF in one place,
- MCP and agent users who want one useful call path before they care about internal plumbing.

## What You Get in 5 Minutes

- Apple Metal aware TensorFlow runtime detection with explicit CPU fallback,
- a 3-layer LSTM + temporal attention forecaster over aligned hydrocarbon / nitrogen / water observables,
- differentiable schedule optimization over the controllable process window,
- lab protocols with GC, FTIR, ion chromatography, yield, and conductivity targets,
- a single `scenario_briefing_tool` payload for MCP clients,
- a clean repo split: `paper/`, `code/`, `data/`, `results/`.

## Repo Layout

```text
paper/    manuscript source and compiled PDF
code/     package, MCP server, scripts, tests
data/     aligned benchmark data and source manifest
results/  model metrics, forecasts, optimization summaries, figures
```

## Fastest Install

Dry run:

```bash
python3 code/scripts/bootstrap_mcp_host.py --venv .venv-tf --dry-run
```

Apple Silicon install:

```bash
python3 code/scripts/bootstrap_mcp_host.py --venv .venv-tf
```

Then verify Metal / CPU fallback:

```bash
.venv-tf/bin/python code/scripts/check_tensorflow_backend.py
```

Then start the MCP server:

```bash
.venv-tf/bin/python -m mcp_server.server
```

Remote HTTP transport (public host):

```bash
HTE_MCP_TRANSPORT=streamable-http FASTMCP_HOST=0.0.0.0 FASTMCP_PORT=8000 .venv-tf/bin/python -m mcp_server.server
```

The default remains `stdio` when `HTE_MCP_TRANSPORT` is not set.

## Remote Deploy (Docker)

1. Keep your private server credentials in `.env` (already gitignored).
2. Shareable template for other users: `.env.example`.
3. Deploy with one command:

```bash
code/scripts/deploy_remote_mcp.sh
```

Template default endpoint path is `/mcp/hormuz` on the port in `HTE_REMOTE_MCP_PORT`.
FastMCP framework default remains `/mcp` when `FASTMCP_STREAMABLE_HTTP_PATH` is not set.
Container restart policy is `unless-stopped`, so it comes back after server reboot by default.

Default anti-spam guard is controlled by `HTE_MCP_MAX_CONCURRENT_REQUESTS=6` (tune in `.env` if needed).

## ChatGPT MCP Usage

After deploy, use this server URL in ChatGPT MCP connector settings:

```text
https://lightcap.ai/mcp/hormuz
```

No password is required by default.

## Manual Install

Apple Silicon:

```bash
/opt/homebrew/bin/python3.11 -m venv .venv-tf
source .venv-tf/bin/activate
pip install -r code/requirements.txt
pip install -r code/requirements-tf-macos.txt
pip install -e './code[dev,tensorflow-apple]'
python code/scripts/check_tensorflow_backend.py
```

Linux / Windows:

```bash
python3 -m venv .venv-tf
source .venv-tf/bin/activate
pip install -r code/requirements.txt
pip install -r code/requirements-tf-linux-windows.txt
pip install -e './code[dev,tensorflow]'
python code/scripts/check_tensorflow_backend.py
```

## TensorFlow Policy

The numerical path is TensorFlow-first.

- If Metal is visible and a probe matmul succeeds, training runs on `GPU:0`.
- If TensorFlow imports but the Metal probe fails, the code records the failure and reroutes training to CPU.
- If TensorFlow is missing, host diagnostics return an explicit install plan instead of silently switching to another framework.

## First Useful Commands

Generate the aligned benchmark dataset:

```bash
.venv-tf/bin/python code/scripts/generate_aligned_dataset.py
```

Rebuild the public artifacts:

```bash
.venv-tf/bin/python code/scripts/rebuild_outputs.py
```

Run the CLI directly:

```bash
.venv-tf/bin/python -m hte.cli rebuild --backend gpu --force-retrain
```

Host diagnostics:

```bash
.venv-tf/bin/python code/scripts/host_doctor.py
```

Tests:

```bash
.venv-tf/bin/python -m pytest code/tests/test_mcp_tools.py -q
```

## MCP Tools

Core surface:

- `backend_status_tool`
- `alignment_manifest_tool`
- `train_model_tool`
- `forecast_observables_tool`
- `optimize_schedule_tool`
- `validation_protocols_tool`
- `write_artifacts_tool`
- `scenario_briefing_tool`
- `host_diagnostics_tool`

`scenario_briefing_tool` is the shortest path from zero to a full payload with backend status, provenance, and generated artifact references.

## Generated Artifacts

Main files produced by the default run:

- `results/model_metrics_lb12_hz1_u128-96-64.json`
- `results/holdout_predictions_lb12_hz1_u128-96-64.csv`
- `results/horizon_forecast_lb12_hz1_u128-96-64.json`
- `results/optimization_summary_lb12_hz1_u128-96-64.json`
- `results/validation_protocols_lb12_hz1_u128-96-64.json`
- `results/figures/holdout_forecast_lb12_hz1_u128-96-64.png`
- `results/figures/optimized_control_schedule_lb12_hz1_u128-96-64.png`
- `paper/paper.pdf`

## Docker

Build:

```bash
docker build -t hte-mcp .
```

Run:

```bash
docker run --rm -it -p 8000:8000 hte-mcp
```

With this repo defaults, use `http://localhost:8000/mcp/hormuz`.
If `FASTMCP_STREAMABLE_HTTP_PATH` is unset, FastMCP default is `http://localhost:8000/mcp`.

## Citation Bridge

- Software metadata: `CITATION.cff`
- License: `LICENSE`
- Manuscript source: `paper/paper.tex`
- Data/source manifest: `data/source_manifest.json`
