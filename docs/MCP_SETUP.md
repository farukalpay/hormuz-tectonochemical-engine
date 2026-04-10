# MCP Setup and Agent Usage

## One-Command Bootstrap

```bash
python3 code/scripts/bootstrap_mcp_host.py --venv .venv-tf
```

Dry run:

```bash
python3 code/scripts/bootstrap_mcp_host.py --venv .venv-tf --dry-run
```

Start the MCP server:

```bash
.venv-tf/bin/python -m mcp_server.server
```

Start MCP over public HTTP:

```bash
HTE_MCP_TRANSPORT=streamable-http FASTMCP_HOST=0.0.0.0 FASTMCP_PORT=8000 .venv-tf/bin/python -m mcp_server.server
```

## Remote Docker Deploy

1. Copy `.env.example` to `.env` and fill your SSH/server values.
2. Run:

```bash
code/scripts/deploy_remote_mcp.sh
```

Template default endpoint path is `/mcp/hormuz`.
FastMCP framework default path is `/mcp` if `FASTMCP_STREAMABLE_HTTP_PATH` is not set.
Container restart policy is `unless-stopped` so it auto-starts after reboot.

Use in ChatGPT MCP connector:

```text
https://lightcap.ai/mcp/hormuz
```

## Host Verification

```bash
.venv-tf/bin/python code/scripts/check_tensorflow_backend.py
.venv-tf/bin/python code/scripts/host_doctor.py
```

## Recommended Tool Order

1. `backend_status_tool`
2. `alignment_manifest_tool`
3. `train_model_tool`
4. `forecast_observables_tool`
5. `optimize_schedule_tool`
6. `validation_protocols_tool`

## Why `scenario_briefing_tool` Exists

`scenario_briefing_tool` bundles backend status, provenance, and artifact paths into one payload for fast MCP sessions.

## Resources

- `hte://alignment/sources`
- `hte://results/latest`

## Logging and Errors

- Every MCP call appends a JSON line to `results/logs/mcp_events.jsonl` when logging is enabled.
- Every tool returns an explicit success or error envelope.
- Host diagnostics report the filesystem layout, backend visibility, and required-file presence.
