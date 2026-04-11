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
Default streamable mode is stateless (`HTE_MCP_STATELESS_HTTP=true`) to prevent session-stickiness failures.
GPU mode is auto-detected on deploy (`HTE_GPU_MODE=auto`) and can be pinned to `nvidia`, `rocm`, or `none`.
TensorFlow distribution is resolved independently (`HTE_TENSORFLOW_DISTRIBUTION=auto|cuda|rocm|cpu`), with `auto` mapping to the resolved GPU mode.
ROCm defaults to `HTE_ROCM_BASE_IMAGE=rocm/tensorflow:latest`; set `HTE_ROCM_HSA_OVERRIDE_GFX_VERSION` only if the host GPU needs an explicit ROCm architecture override.
Use `HTE_DOCKER_BASE_IMAGE` to pin a custom base image for any mode.
Set `HTE_REQUIRE_GPU=true` to fail deployment when backend probe cannot resolve `GPU:0`.
The runtime disables TensorFlow XLA JIT and enables GPU memory growth before probing or training to avoid unstable ROCm autotune paths.
Training determinism defaults to `auto`: CPU runs use deterministic ops, ROCm GPU runs keep seeded execution without forcing the determinism path that breaks training.
When `GPU:0` becomes healthy again, cached CPU-fallback models are retrained instead of being silently reused.
Default OAuth consent is approve-only unless `HTE_OAUTH_APPROVAL_PASSWORD_HASH` is configured.
Set `HTE_OAUTH_PUBLIC_BASE_URL` to your HTTPS domain for stable OAuth metadata URLs.
`HTE_OAUTH_PUBLIC_BASE_URL` and `HTE_ARTIFACT_PUBLIC_BASE_URL` should use `https://`.
Authorize UI endpoint is `<base-url>/mcp/hormuz/authorize`.
Artifact links are enabled by default and served from `<base-url>/mcp/hormuz/artifacts/...`.
Disable with `HTE_ARTIFACT_LINKS_ENABLED=false` if a host does not want public artifact URLs.

Use in ChatGPT MCP connector:

```text
https://lightcap.ai/mcp/hormuz
```

Generate optional consent password hash:

```bash
python code/scripts/generate_oauth_approval_password_hash.py
```

`/mcp/nexus` remains a separate gateway with its own credential-binding flow.

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
