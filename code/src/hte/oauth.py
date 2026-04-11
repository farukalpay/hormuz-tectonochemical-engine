from __future__ import annotations

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

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response


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


def load_oauth_ui_config() -> OAuthUiConfig:
    password_hash = os.environ.get("HTE_OAUTH_APPROVAL_PASSWORD_HASH", "").strip() or None
    public_base_url = os.environ.get("HTE_OAUTH_PUBLIC_BASE_URL", "").strip() or None
    if public_base_url:
        public_base_url = public_base_url.rstrip("/")
    title = os.environ.get("HTE_OAUTH_PAGE_TITLE", "Authorize MCP Access").strip() or "Authorize MCP Access"
    subtitle = (
        os.environ.get("HTE_OAUTH_PAGE_SUBTITLE", "Review and approve this MCP client request.").strip()
        or "Review and approve this MCP client request."
    )
    return OAuthUiConfig(
        approval_password_hash=password_hash,
        public_base_url=public_base_url,
        page_title=title,
        page_subtitle=subtitle,
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


class OAuthUiServer:
    def __init__(self, resource_path: str, config: OAuthUiConfig | None = None) -> None:
        self._resource_path = resource_path
        self._config = config or load_oauth_ui_config()
        self._clients: dict[str, OAuthClient] = {}
        self._requests: dict[str, OAuthAuthorizationRequest] = {}
        self._codes: dict[str, OAuthAuthorizationCode] = {}
        template_path = Path(__file__).resolve().parent / "web" / "authorize.html"
        self._template = template_path.read_text(encoding="utf-8")

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
            key: value for key, value in self._requests.items() if now - value.created_at <= 600
        }
        self._codes = {
            key: value for key, value in self._codes.items() if value.expires_at > now
        }

    async def handle_register(self, request: Request) -> Response:
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
        self._purge_expired()
        params = request.query_params
        if params.get("response_type") != "code":
            return JSONResponse({"error": "unsupported_response_type"}, status_code=400)
        client_id = params.get("client_id", "")
        redirect_uri = params.get("redirect_uri", "")
        client = self._clients.get(client_id)
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
        self._purge_expired()
        form = await request.form()
        request_id = str(form.get("request_id") or "")
        decision = str(form.get("decision") or "deny").lower()
        auth_request = self._requests.get(request_id)
        if auth_request is None:
            return JSONResponse({"error": "invalid_request", "error_description": "request expired"}, status_code=410)
        client = self._clients.get(auth_request.client_id)
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
            expires_at=time.time() + 300,
        )
        self._requests.pop(request_id, None)
        query = {"code": code}
        if auth_request.state:
            query["state"] = auth_request.state
        separator = "&" if "?" in auth_request.redirect_uri else "?"
        return RedirectResponse(f"{auth_request.redirect_uri}{separator}{urlencode(query)}", status_code=302)

    async def handle_token(self, request: Request) -> Response:
        self._purge_expired()
        form = await request.form()
        form_data = {key: str(value) for key, value in form.items()}
        if form_data.get("grant_type") != "authorization_code":
            return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)

        code = form_data.get("code", "")
        auth_code = self._codes.get(code)
        if auth_code is None:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        client = self._clients.get(auth_code.client_id)
        if client is None:
            return JSONResponse({"error": "invalid_client"}, status_code=401)
        redirect_uri = form_data.get("redirect_uri", "")
        if redirect_uri != auth_code.redirect_uri:
            return JSONResponse({"error": "invalid_grant", "error_description": "redirect_uri mismatch"}, status_code=400)
        if form_data.get("client_id", "") != client.client_id:
            return JSONResponse({"error": "invalid_client"}, status_code=401)

        if client.token_endpoint_auth_method in {"client_secret_post", "client_secret_basic"}:
            provided_secret = _client_secret_from_request(request, form_data)
            if not client.client_secret or not provided_secret or not hmac.compare_digest(client.client_secret, provided_secret):
                return JSONResponse({"error": "invalid_client"}, status_code=401)

        code_verifier = form_data.get("code_verifier", "")
        if not _pkce_valid(code_verifier, auth_code.code_challenge, auth_code.code_challenge_method):
            return JSONResponse({"error": "invalid_grant", "error_description": "PKCE verification failed"}, status_code=400)

        self._codes.pop(code, None)
        return JSONResponse(
            {
                "access_token": secrets.token_urlsafe(32),
                "token_type": "Bearer",
                "expires_in": 3600,
                "refresh_token": secrets.token_urlsafe(32),
                "scope": auth_code.scope,
            }
        )

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
