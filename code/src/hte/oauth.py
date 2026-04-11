from __future__ import annotations

import asyncio
import base64
import html
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.parse import urlparse

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response

from .paths import RESULTS_ROOT

DEFAULT_ACCESS_TOKEN_TTL_SECONDS = 3600
DEFAULT_REFRESH_TOKEN_TTL_SECONDS = 30 * 24 * 60 * 60
DEFAULT_AUTHORIZATION_CODE_TTL_SECONDS = 300
DEFAULT_AUTH_REQUEST_TTL_SECONDS = 600


def _b64url_sha256(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _pkce_valid(code_verifier: str, code_challenge: str | None, method: str | None) -> bool:
    if not code_challenge:
        return True
    challenge_method = (method or "plain").lower()
    if challenge_method == "plain":
        return hmac.compare_digest(code_verifier, code_challenge)
    if challenge_method == "s256":
        return hmac.compare_digest(_b64url_sha256(code_verifier), code_challenge)
    return False


def _client_secret_from_request(request: Request, form_data: dict[str, str]) -> str | None:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("basic "):
        payload = header[6:].strip()
        try:
            decoded = base64.b64decode(payload).decode("utf-8")
        except Exception:
            return None
        if ":" in decoded:
            return decoded.split(":", 1)[1]
    return form_data.get("client_secret")


def hash_approval_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    salt_b64 = base64.urlsafe_b64encode(salt).decode("ascii").rstrip("=")
    digest_b64 = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"scrypt$v1${salt_b64}${digest_b64}"


def verify_approval_password(password: str, encoded_hash: str) -> bool:
    try:
        algo, version, salt_b64, digest_b64 = encoded_hash.split("$", 3)
        if algo != "scrypt" or version != "v1":
            return False
        salt = base64.urlsafe_b64decode(salt_b64 + "=" * (-len(salt_b64) % 4))
        expected = base64.urlsafe_b64decode(digest_b64 + "=" * (-len(digest_b64) % 4))
    except Exception:
        return False
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return hmac.compare_digest(digest, expected)


@dataclass(frozen=True)
class OAuthUiConfig:
    approval_password_hash: str | None = None
    public_base_url: str | None = None
    page_title: str = "Authorize MCP Access"
    page_subtitle: str = "Review and approve this MCP client request."
    state_file: Path | None = None
    access_token_ttl_seconds: int = DEFAULT_ACCESS_TOKEN_TTL_SECONDS
    refresh_token_ttl_seconds: int = DEFAULT_REFRESH_TOKEN_TTL_SECONDS
    authorization_code_ttl_seconds: int = DEFAULT_AUTHORIZATION_CODE_TTL_SECONDS


def load_oauth_ui_config() -> OAuthUiConfig:
    def _read_positive_int(name: str, default: int) -> int:
        raw = os.environ.get(name, "").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a positive integer") from exc
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    password_hash = os.environ.get("HTE_OAUTH_APPROVAL_PASSWORD_HASH", "").strip() or None
    public_base_url = os.environ.get("HTE_OAUTH_PUBLIC_BASE_URL", "").strip() or None
    if public_base_url:
        parsed = urlparse(public_base_url)
        if parsed.scheme.lower() != "https" or not parsed.netloc:
            raise ValueError("HTE_OAUTH_PUBLIC_BASE_URL must be an absolute https URL")
        public_base_url = public_base_url.rstrip("/")
    title = os.environ.get("HTE_OAUTH_PAGE_TITLE", "Authorize MCP Access").strip() or "Authorize MCP Access"
    subtitle = (
        os.environ.get("HTE_OAUTH_PAGE_SUBTITLE", "Review and approve this MCP client request.").strip()
        or "Review and approve this MCP client request."
    )
    state_file_raw = os.environ.get("HTE_OAUTH_STATE_FILE", "").strip()
    state_file = Path(state_file_raw).expanduser() if state_file_raw else RESULTS_ROOT / "state" / "oauth_state.json"
    access_token_ttl_seconds = _read_positive_int("HTE_OAUTH_ACCESS_TOKEN_TTL_SECONDS", DEFAULT_ACCESS_TOKEN_TTL_SECONDS)
    refresh_token_ttl_seconds = _read_positive_int("HTE_OAUTH_REFRESH_TOKEN_TTL_SECONDS", DEFAULT_REFRESH_TOKEN_TTL_SECONDS)
    authorization_code_ttl_seconds = _read_positive_int(
        "HTE_OAUTH_AUTHORIZATION_CODE_TTL_SECONDS",
        DEFAULT_AUTHORIZATION_CODE_TTL_SECONDS,
    )
    return OAuthUiConfig(
        approval_password_hash=password_hash,
        public_base_url=public_base_url,
        page_title=title,
        page_subtitle=subtitle,
        state_file=state_file,
        access_token_ttl_seconds=access_token_ttl_seconds,
        refresh_token_ttl_seconds=refresh_token_ttl_seconds,
        authorization_code_ttl_seconds=authorization_code_ttl_seconds,
    )


@dataclass(frozen=True)
class OAuthClient:
    client_id: str
    redirect_uris: tuple[str, ...]
    token_endpoint_auth_method: str
    client_secret: str | None
    client_name: str
    scope: str = "mcp"


@dataclass
class OAuthAuthorizationRequest:
    request_id: str
    client_id: str
    redirect_uri: str
    state: str | None
    scope: str
    code_challenge: str | None
    code_challenge_method: str | None
    created_at: float = field(default_factory=time.time)


@dataclass
class OAuthAuthorizationCode:
    code: str
    client_id: str
    redirect_uri: str
    scope: str
    code_challenge: str | None
    code_challenge_method: str | None
    expires_at: float


@dataclass
class OAuthRefreshToken:
    token: str
    client_id: str
    scope: str
    expires_at: float


class OAuthUiServer:
    def __init__(self, resource_path: str, config: OAuthUiConfig | None = None) -> None:
        self._resource_path = resource_path
        self._config = config or load_oauth_ui_config()
        self._state_lock = asyncio.Lock()
        self._clients: dict[str, OAuthClient] = {}
        self._requests: dict[str, OAuthAuthorizationRequest] = {}
        self._codes: dict[str, OAuthAuthorizationCode] = {}
        self._refresh_tokens: dict[str, OAuthRefreshToken] = {}
        template_path = Path(__file__).resolve().parent / "web" / "authorize.html"
        self._template = template_path.read_text(encoding="utf-8")
        self._load_state_from_disk()

    def _scope(self) -> str:
        return self._resource_path.strip("/").replace("/", ":")

    def _origin(self, request: Request) -> str:
        if self._config.public_base_url:
            return self._config.public_base_url

        scheme = request.url.scheme
        cf_visitor = request.headers.get("cf-visitor", "")
        if cf_visitor:
            try:
                parsed_visitor = json.loads(cf_visitor)
                visitor_scheme = str(parsed_visitor.get("scheme") or "").strip().lower()
                if visitor_scheme in {"http", "https"}:
                    scheme = visitor_scheme
            except Exception:
                pass

        forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",", 1)[0].strip().lower()
        if forwarded_proto in {"http", "https"}:
            scheme = forwarded_proto

        host = request.headers.get("x-forwarded-host", "").split(",", 1)[0].strip()
        if not host:
            host = request.headers.get("host", "").strip()
        if host:
            return f"{scheme}://{host}"

        return str(request.base_url).rstrip("/")

    def _full_url(self, request: Request, path: str) -> str:
        return f"{self._origin(request)}{path}"

    def _redirect_with_error(self, redirect_uri: str, error: str, state: str | None, description: str | None = None) -> RedirectResponse:
        query = {"error": error}
        if description:
            query["error_description"] = description
        if state:
            query["state"] = state
        separator = "&" if "?" in redirect_uri else "?"
        return RedirectResponse(f"{redirect_uri}{separator}{urlencode(query)}", status_code=302)

    def _render_authorize_page(
        self,
        auth_request: OAuthAuthorizationRequest,
        client: OAuthClient,
        request: Request,
        error_message: str | None = None,
    ) -> HTMLResponse:
        password_field = ""
        if self._config.approval_password_hash:
            password_field = (
                '<label><span>Approval Password</span>'
                '<input type="password" name="approval_password" autocomplete="current-password" required /></label>'
            )
        error_block = (
            f'<div class="error">{html.escape(error_message, quote=True)}</div>'
            if error_message
            else ""
        )
        rendered_html = (
            self._template.replace("{{PAGE_TITLE}}", html.escape(self._config.page_title, quote=True))
            .replace("{{PAGE_SUBTITLE}}", html.escape(self._config.page_subtitle, quote=True))
            .replace("{{RESOURCE_URI}}", html.escape(self._full_url(request, self._resource_path), quote=True))
            .replace("{{SCOPES}}", html.escape(auth_request.scope, quote=True))
            .replace("{{CLIENT_NAME}}", html.escape(client.client_name, quote=True))
            .replace("{{REQUEST_ID}}", html.escape(auth_request.request_id, quote=True))
            .replace("{{CONSENT_ACTION}}", html.escape(f"{self._resource_path}/consent", quote=True))
            .replace("{{PASSWORD_FIELD}}", password_field)
            .replace("{{ERROR_BLOCK}}", error_block)
        )
        return HTMLResponse(rendered_html)

    def _purge_expired(self) -> None:
        now = time.time()
        self._requests = {
            key: value for key, value in self._requests.items() if now - value.created_at <= DEFAULT_AUTH_REQUEST_TTL_SECONDS
        }
        self._codes = {
            key: value for key, value in self._codes.items() if value.expires_at > now
        }
        changed = False
        for token, refresh_token in list(self._refresh_tokens.items()):
            if refresh_token.expires_at <= now:
                self._refresh_tokens.pop(token, None)
                changed = True
        if changed:
            self._persist_state()

    def _state_file_path(self) -> Path | None:
        return self._config.state_file

    @staticmethod
    def _client_from_payload(client_id: str, payload: object) -> OAuthClient | None:
        if not isinstance(payload, dict):
            return None
        redirect_uris = payload.get("redirect_uris")
        if not isinstance(redirect_uris, list):
            return None
        token_method = str(payload.get("token_endpoint_auth_method") or "none")
        if token_method not in {"none", "client_secret_post", "client_secret_basic"}:
            return None
        return OAuthClient(
            client_id=client_id,
            redirect_uris=tuple(str(uri) for uri in redirect_uris if str(uri)),
            token_endpoint_auth_method=token_method,
            client_secret=str(payload.get("client_secret")) if payload.get("client_secret") is not None else None,
            client_name=str(payload.get("client_name") or "HTE MCP Client"),
            scope=str(payload.get("scope") or "mcp"),
        )

    @staticmethod
    def _refresh_token_from_payload(token: str, payload: object) -> OAuthRefreshToken | None:
        if not isinstance(payload, dict):
            return None
        client_id = str(payload.get("client_id") or "")
        scope = str(payload.get("scope") or "")
        expires_at_raw = payload.get("expires_at")
        try:
            expires_at = float(expires_at_raw)
        except (TypeError, ValueError):
            return None
        if not client_id or not scope:
            return None
        return OAuthRefreshToken(
            token=token,
            client_id=client_id,
            scope=scope,
            expires_at=expires_at,
        )

    def _load_state_from_disk(self) -> None:
        state_path = self._state_file_path()
        if state_path is None or not state_path.exists():
            return
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        clients_payload = payload.get("clients")
        refresh_tokens_payload = payload.get("refresh_tokens")
        loaded_clients: dict[str, OAuthClient] = {}
        loaded_refresh_tokens: dict[str, OAuthRefreshToken] = {}
        if isinstance(clients_payload, dict):
            for client_id, item in clients_payload.items():
                parsed = self._client_from_payload(str(client_id), item)
                if parsed is not None and parsed.redirect_uris:
                    loaded_clients[parsed.client_id] = parsed
        if isinstance(refresh_tokens_payload, dict):
            now = time.time()
            for token, item in refresh_tokens_payload.items():
                parsed = self._refresh_token_from_payload(str(token), item)
                if parsed is not None and parsed.expires_at > now:
                    loaded_refresh_tokens[parsed.token] = parsed
        if loaded_clients:
            self._clients.update(loaded_clients)
        if loaded_refresh_tokens:
            self._refresh_tokens.update(loaded_refresh_tokens)

    def _persist_state(self) -> None:
        state_path = self._state_file_path()
        if state_path is None:
            return
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": int(time.time()),
            "clients": {
                client_id: {
                    "redirect_uris": list(client.redirect_uris),
                    "token_endpoint_auth_method": client.token_endpoint_auth_method,
                    "client_secret": client.client_secret,
                    "client_name": client.client_name,
                    "scope": client.scope,
                }
                for client_id, client in self._clients.items()
            },
            "refresh_tokens": {
                token: {
                    "client_id": refresh_token.client_id,
                    "scope": refresh_token.scope,
                    "expires_at": refresh_token.expires_at,
                }
                for token, refresh_token in self._refresh_tokens.items()
            },
        }
        temp_path = state_path.with_suffix(state_path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True), encoding="utf-8")
        temp_path.replace(state_path)

    def _lookup_client(self, client_id: str) -> OAuthClient | None:
        client = self._clients.get(client_id)
        if client is not None:
            return client
        self._load_state_from_disk()
        return self._clients.get(client_id)

    def _lookup_refresh_token(self, refresh_token: str) -> OAuthRefreshToken | None:
        token = self._refresh_tokens.get(refresh_token)
        if token is not None:
            return token
        self._load_state_from_disk()
        return self._refresh_tokens.get(refresh_token)

    @staticmethod
    def _scope_within(requested_scope: str, granted_scope: str) -> bool:
        requested = {item for item in requested_scope.split() if item}
        granted = {item for item in granted_scope.split() if item}
        return requested.issubset(granted)

    def _issue_token_pair(self, *, client_id: str, scope: str, rotate_from: str | None = None) -> dict[str, Any]:
        now = time.time()
        refresh_token = secrets.token_urlsafe(40)
        refresh_record = OAuthRefreshToken(
            token=refresh_token,
            client_id=client_id,
            scope=scope,
            expires_at=now + self._config.refresh_token_ttl_seconds,
        )
        if rotate_from:
            self._refresh_tokens.pop(rotate_from, None)
        self._refresh_tokens[refresh_token] = refresh_record
        self._persist_state()
        return {
            "access_token": secrets.token_urlsafe(32),
            "token_type": "Bearer",
            "expires_in": self._config.access_token_ttl_seconds,
            "refresh_token": refresh_token,
            "scope": scope,
        }

    def _validate_client_auth(
        self,
        *,
        request: Request,
        form_data: dict[str, str],
        client: OAuthClient,
    ) -> Response | None:
        if form_data.get("client_id", "") != client.client_id:
            return JSONResponse({"error": "invalid_client"}, status_code=401)
        if client.token_endpoint_auth_method in {"client_secret_post", "client_secret_basic"}:
            provided_secret = _client_secret_from_request(request, form_data)
            if not client.client_secret or not provided_secret or not hmac.compare_digest(client.client_secret, provided_secret):
                return JSONResponse({"error": "invalid_client"}, status_code=401)
        return None

    async def handle_register(self, request: Request) -> Response:
        async with self._state_lock:
            payload = await request.json()
            redirect_uris = payload.get("redirect_uris") or []
            if not isinstance(redirect_uris, list) or not redirect_uris:
                return JSONResponse({"error": "invalid_client_metadata", "error_description": "redirect_uris is required"}, status_code=400)
            token_method = payload.get("token_endpoint_auth_method", "none")
            if token_method not in {"none", "client_secret_post", "client_secret_basic"}:
                return JSONResponse(
                    {"error": "invalid_client_metadata", "error_description": "unsupported token_endpoint_auth_method"},
                    status_code=400,
                )
            client_id = secrets.token_urlsafe(18)
            client_secret = secrets.token_urlsafe(32) if token_method != "none" else None
            client_name = str(payload.get("client_name") or "HTE MCP Client")
            scope = self._scope()
            client = OAuthClient(
                client_id=client_id,
                redirect_uris=tuple(str(uri) for uri in redirect_uris),
                token_endpoint_auth_method=token_method,
                client_secret=client_secret,
                client_name=client_name,
                scope=scope,
            )
            self._clients[client_id] = client
            self._persist_state()
            response_payload: dict[str, Any] = {
                "client_id": client.client_id,
                "redirect_uris": list(client.redirect_uris),
                "token_endpoint_auth_method": client.token_endpoint_auth_method,
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "scope": client.scope,
                "client_name": client.client_name,
                "client_id_issued_at": int(time.time()),
            }
            if client_secret:
                response_payload["client_secret"] = client_secret
            return JSONResponse(response_payload)

    async def handle_authorize(self, request: Request) -> Response:
        async with self._state_lock:
            self._purge_expired()
            params = request.query_params
            if params.get("response_type") != "code":
                return JSONResponse({"error": "unsupported_response_type"}, status_code=400)
            client_id = params.get("client_id", "")
            redirect_uri = params.get("redirect_uri", "")
            client = self._lookup_client(client_id)
            if client is None:
                return JSONResponse({"error": "invalid_request", "error_description": "unknown client_id"}, status_code=400)
            if redirect_uri not in client.redirect_uris:
                return JSONResponse({"error": "invalid_request", "error_description": "redirect_uri mismatch"}, status_code=400)
            request_scope = params.get("scope") or client.scope
            request_id = secrets.token_urlsafe(18)
            auth_request = OAuthAuthorizationRequest(
                request_id=request_id,
                client_id=client_id,
                redirect_uri=redirect_uri,
                state=params.get("state"),
                scope=request_scope,
                code_challenge=params.get("code_challenge"),
                code_challenge_method=params.get("code_challenge_method"),
            )
            self._requests[request_id] = auth_request
            return self._render_authorize_page(auth_request=auth_request, client=client, request=request)

    async def handle_consent(self, request: Request) -> Response:
        async with self._state_lock:
            self._purge_expired()
            form = await request.form()
            request_id = str(form.get("request_id") or "")
            decision = str(form.get("decision") or "deny").lower()
            auth_request = self._requests.get(request_id)
            if auth_request is None:
                return JSONResponse({"error": "invalid_request", "error_description": "request expired"}, status_code=410)
            client = self._lookup_client(auth_request.client_id)
            if client is None:
                return JSONResponse({"error": "invalid_request", "error_description": "unknown client"}, status_code=400)
            if decision != "approve":
                self._requests.pop(request_id, None)
                return self._redirect_with_error(auth_request.redirect_uri, "access_denied", auth_request.state)

            if self._config.approval_password_hash:
                provided = str(form.get("approval_password") or "")
                if not verify_approval_password(provided, self._config.approval_password_hash):
                    return self._render_authorize_page(
                        auth_request=auth_request,
                        client=client,
                        request=request,
                        error_message="Approval password is incorrect.",
                    )

            code = secrets.token_urlsafe(24)
            self._codes[code] = OAuthAuthorizationCode(
                code=code,
                client_id=auth_request.client_id,
                redirect_uri=auth_request.redirect_uri,
                scope=auth_request.scope,
                code_challenge=auth_request.code_challenge,
                code_challenge_method=auth_request.code_challenge_method,
                expires_at=time.time() + self._config.authorization_code_ttl_seconds,
            )
            self._requests.pop(request_id, None)
            query = {"code": code}
            if auth_request.state:
                query["state"] = auth_request.state
            separator = "&" if "?" in auth_request.redirect_uri else "?"
            return RedirectResponse(f"{auth_request.redirect_uri}{separator}{urlencode(query)}", status_code=302)

    async def handle_token(self, request: Request) -> Response:
        async with self._state_lock:
            self._purge_expired()
            form = await request.form()
            form_data = {key: str(value) for key, value in form.items()}
            grant_type = form_data.get("grant_type", "")
            if grant_type == "authorization_code":
                code = form_data.get("code", "")
                auth_code = self._codes.get(code)
                if auth_code is None:
                    return JSONResponse({"error": "invalid_grant"}, status_code=400)
                client = self._lookup_client(auth_code.client_id)
                if client is None:
                    return JSONResponse({"error": "invalid_client"}, status_code=401)
                redirect_uri = form_data.get("redirect_uri", "")
                if redirect_uri != auth_code.redirect_uri:
                    return JSONResponse({"error": "invalid_grant", "error_description": "redirect_uri mismatch"}, status_code=400)
                auth_error = self._validate_client_auth(request=request, form_data=form_data, client=client)
                if auth_error is not None:
                    return auth_error
                code_verifier = form_data.get("code_verifier", "")
                if not _pkce_valid(code_verifier, auth_code.code_challenge, auth_code.code_challenge_method):
                    return JSONResponse({"error": "invalid_grant", "error_description": "PKCE verification failed"}, status_code=400)
                self._codes.pop(code, None)
                return JSONResponse(self._issue_token_pair(client_id=client.client_id, scope=auth_code.scope))
            if grant_type == "refresh_token":
                refresh_token = form_data.get("refresh_token", "")
                if not refresh_token:
                    return JSONResponse({"error": "invalid_grant"}, status_code=400)
                client_id = form_data.get("client_id", "")
                client = self._lookup_client(client_id)
                if client is None:
                    return JSONResponse({"error": "invalid_client"}, status_code=401)
                auth_error = self._validate_client_auth(request=request, form_data=form_data, client=client)
                if auth_error is not None:
                    return auth_error
                refresh_record = self._lookup_refresh_token(refresh_token)
                if refresh_record is None or refresh_record.expires_at <= time.time():
                    self._refresh_tokens.pop(refresh_token, None)
                    self._persist_state()
                    return JSONResponse({"error": "invalid_grant"}, status_code=400)
                if refresh_record.client_id != client.client_id:
                    return JSONResponse({"error": "invalid_grant"}, status_code=400)
                requested_scope = form_data.get("scope", "").strip() or refresh_record.scope
                if not self._scope_within(requested_scope, refresh_record.scope):
                    return JSONResponse({"error": "invalid_scope"}, status_code=400)
                return JSONResponse(
                    self._issue_token_pair(
                        client_id=client.client_id,
                        scope=requested_scope,
                        rotate_from=refresh_token,
                    )
                )
            return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

    async def handle_authorization_server_metadata(self, request: Request) -> Response:
        base = self._origin(request)
        return JSONResponse(
            {
                "issuer": f"{base}{self._resource_path}",
                "authorization_endpoint": f"{base}{self._resource_path}/authorize",
                "token_endpoint": f"{base}{self._resource_path}/token",
                "registration_endpoint": f"{base}{self._resource_path}/register",
                "response_types_supported": ["code"],
                "grant_types_supported": ["authorization_code", "refresh_token"],
                "token_endpoint_auth_methods_supported": ["none", "client_secret_post", "client_secret_basic"],
                "code_challenge_methods_supported": ["S256", "plain"],
                "scopes_supported": [self._scope()],
            }
        )

    async def handle_protected_resource_metadata(self, request: Request) -> Response:
        base = self._origin(request)
        return JSONResponse(
            {
                "resource": f"{base}{self._resource_path}",
                "authorization_servers": [f"{base}{self._resource_path}"],
                "scopes_supported": [self._scope()],
                "bearer_methods_supported": ["header"],
            }
        )
