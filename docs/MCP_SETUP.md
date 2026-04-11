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

### Runtime Defaults

- Template endpoint path defaults to `/mcp/hormuz`.
- FastMCP default streamable path is `/mcp` when `FASTMCP_STREAMABLE_HTTP_PATH` is unset.
- Container restart policy is `unless-stopped` (auto-start after reboot).
- Streamable mode defaults to stateless (`HTE_MCP_STATELESS_HTTP=true`) to avoid session-stickiness failures.

### GPU and TensorFlow Controls

- GPU mode is auto-detected on deploy (`HTE_GPU_MODE=auto`); pin with `nvidia`, `rocm`, or `none`.
- TensorFlow distribution is resolved independently (`HTE_TENSORFLOW_DISTRIBUTION=auto|cuda|rocm|cpu`), where `auto` maps to the resolved GPU mode.
- ROCm base image defaults to `HTE_ROCM_BASE_IMAGE=rocm/tensorflow:latest`.
- Set `HTE_ROCM_HSA_OVERRIDE_GFX_VERSION` only when your ROCm host GPU needs an explicit architecture override.
- Use `HTE_DOCKER_BASE_IMAGE` to pin a custom base image for any mode.
- Set `HTE_REQUIRE_GPU=true` to fail deployment if backend probe cannot resolve `GPU:0`; at runtime, GPU-requested training/forecast calls also fail instead of dropping to CPU.
- Runtime disables TensorFlow XLA JIT.
- Runtime enables GPU memory growth before probing/training to avoid unstable ROCm autotune paths.
- Determinism defaults to `auto`: CPU uses deterministic ops; ROCm GPU keeps seeded execution without the determinism path that breaks training.
- When `GPU:0` becomes healthy again, cached CPU-fallback models are retrained rather than silently reused.

### OAuth and Artifact URLs

- OAuth consent defaults to approve-only unless `HTE_OAUTH_APPROVAL_PASSWORD_HASH` is configured.
- Set `HTE_OAUTH_PUBLIC_BASE_URL` to your HTTPS domain for stable OAuth metadata URLs.
- `HTE_OAUTH_PUBLIC_BASE_URL` and `HTE_ARTIFACT_PUBLIC_BASE_URL` should use `https://`.
- OAuth clients and refresh tokens persist to `HTE_OAUTH_STATE_FILE` (default `/app/results/state/oauth_state.json`).
- Keep `/app/results` on a persistent host mount to avoid `unknown client_id` after container recreation.
- Token lifetimes can be tuned with `HTE_OAUTH_ACCESS_TOKEN_TTL_SECONDS`, `HTE_OAUTH_REFRESH_TOKEN_TTL_SECONDS`, and `HTE_OAUTH_AUTHORIZATION_CODE_TTL_SECONDS`.
- Authorize UI endpoint: `<base-url>/mcp/hormuz/authorize`.
- Artifact links are enabled by default and served from `<base-url>/mcp/hormuz/artifacts/...`.
- Disable artifact links with `HTE_ARTIFACT_LINKS_ENABLED=false`.

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
