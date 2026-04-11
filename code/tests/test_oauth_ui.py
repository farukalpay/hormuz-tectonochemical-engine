from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from hte.oauth import OAuthUiConfig, OAuthUiServer, hash_approval_password, load_oauth_ui_config, verify_approval_password


@pytest.fixture(autouse=True)
def _isolated_oauth_state_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HTE_OAUTH_STATE_FILE", str(tmp_path / "oauth_state.json"))


def _app_for(server: OAuthUiServer) -> Starlette:
    return Starlette(
        routes=[
            Route("/mcp/hormuz/register", server.handle_register, methods=["POST"]),
            Route("/mcp/hormuz/authorize", server.handle_authorize, methods=["GET"]),
            Route("/mcp/hormuz/consent", server.handle_consent, methods=["POST"]),
            Route("/mcp/hormuz/token", server.handle_token, methods=["POST"]),
            Route(
                "/.well-known/oauth-authorization-server/mcp/hormuz",
                server.handle_authorization_server_metadata,
                methods=["GET"],
            ),
            Route(
                "/.well-known/oauth-protected-resource/mcp/hormuz",
                server.handle_protected_resource_metadata,
                methods=["GET"],
            ),
        ]
    )


def _extract_request_id(html: str) -> str:
    match = re.search(r'name="request_id" value="([^"]+)"', html)
    assert match is not None
    return match.group(1)


def test_password_hash_roundtrip() -> None:
    encoded = hash_approval_password("top-secret")
    assert verify_approval_password("top-secret", encoded) is True
    assert verify_approval_password("wrong", encoded) is False


def test_oauth_ui_approve_flow_without_password() -> None:
    server = OAuthUiServer(resource_path="/mcp/hormuz")
    client = TestClient(_app_for(server))

    register = client.post(
        "/mcp/hormuz/register",
        json={
            "client_name": "Test Client",
            "redirect_uris": ["https://example.com/callback"],
            "token_endpoint_auth_method": "none",
        },
    )
    assert register.status_code == 200
    payload = register.json()

    authorize = client.get(
        "/mcp/hormuz/authorize",
        params={
            "response_type": "code",
            "client_id": payload["client_id"],
            "redirect_uri": "https://example.com/callback",
            "scope": payload["scope"],
            "state": "abc123",
            "code_challenge": "pkce-verifier-works",
            "code_challenge_method": "plain",
        },
    )
    assert authorize.status_code == 200
    assert 'value="approve"' in authorize.text
    assert "Cancel" not in authorize.text
    request_id = _extract_request_id(authorize.text)

    consent = client.post(
        "/mcp/hormuz/consent",
        data={"request_id": request_id, "decision": "approve"},
        follow_redirects=False,
    )
    assert consent.status_code == 302
    redirect_location = consent.headers["location"]
    parsed = urlparse(redirect_location)
    query = parse_qs(parsed.query)
    assert parsed.scheme == "https"
    assert parsed.netloc == "example.com"
    assert "code" in query
    assert query["state"][0] == "abc123"

    token = client.post(
        "/mcp/hormuz/token",
        data={
            "grant_type": "authorization_code",
            "client_id": payload["client_id"],
            "redirect_uri": "https://example.com/callback",
            "code": query["code"][0],
            "code_verifier": "pkce-verifier-works",
        },
    )
    assert token.status_code == 200
    assert "access_token" in token.json()


def test_oauth_ui_password_gate() -> None:
    config = OAuthUiConfig(
        approval_password_hash=hash_approval_password("letmein"),
        page_title="Authorize",
        page_subtitle="Approve request",
    )
    server = OAuthUiServer(resource_path="/mcp/hormuz", config=config)
    client = TestClient(_app_for(server))

    register = client.post(
        "/mcp/hormuz/register",
        json={
            "client_name": "Secured Client",
            "redirect_uris": ["https://example.com/callback"],
            "token_endpoint_auth_method": "none",
        },
    )
    payload = register.json()

    authorize = client.get(
        "/mcp/hormuz/authorize",
        params={
            "response_type": "code",
            "client_id": payload["client_id"],
            "redirect_uri": "https://example.com/callback",
            "scope": payload["scope"],
        },
    )
    request_id = _extract_request_id(authorize.text)

    deny = client.post(
        "/mcp/hormuz/consent",
        data={"request_id": request_id, "decision": "approve", "approval_password": "wrong"},
    )
    assert deny.status_code == 200
    assert "Approval password is incorrect." in deny.text

    approve = client.post(
        "/mcp/hormuz/consent",
        data={"request_id": request_id, "decision": "approve", "approval_password": "letmein"},
        follow_redirects=False,
    )
    assert approve.status_code == 302


def test_oauth_metadata_uses_public_base_url() -> None:
    config = OAuthUiConfig(
        public_base_url="https://lightcap.ai",
        page_title="Authorize",
        page_subtitle="Approve request",
    )
    server = OAuthUiServer(resource_path="/mcp/hormuz", config=config)
    client = TestClient(_app_for(server))

    authorization_server = client.get("/.well-known/oauth-authorization-server/mcp/hormuz")
    assert authorization_server.status_code == 200
    payload = authorization_server.json()
    assert payload["issuer"] == "https://lightcap.ai/mcp/hormuz"
    assert payload["authorization_endpoint"] == "https://lightcap.ai/mcp/hormuz/authorize"
    assert payload["token_endpoint"] == "https://lightcap.ai/mcp/hormuz/token"

    protected_resource = client.get("/.well-known/oauth-protected-resource/mcp/hormuz")
    assert protected_resource.status_code == 200
    protected_payload = protected_resource.json()
    assert protected_payload["resource"] == "https://lightcap.ai/mcp/hormuz"


def test_public_base_url_requires_https(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTE_OAUTH_PUBLIC_BASE_URL", "http://lightcap.ai")
    with pytest.raises(ValueError, match="must be an absolute https URL"):
        load_oauth_ui_config()


def test_oauth_client_registry_survives_server_restart(tmp_path: Path) -> None:
    state_file = tmp_path / "oauth_state.json"
    config = OAuthUiConfig(state_file=state_file)
    server = OAuthUiServer(resource_path="/mcp/hormuz", config=config)
    client = TestClient(_app_for(server))

    register = client.post(
        "/mcp/hormuz/register",
        json={
            "client_name": "Persistent Client",
            "redirect_uris": ["https://example.com/callback"],
            "token_endpoint_auth_method": "none",
        },
    )
    assert register.status_code == 200
    payload = register.json()

    reloaded_server = OAuthUiServer(resource_path="/mcp/hormuz", config=config)
    reloaded_client = TestClient(_app_for(reloaded_server))
    authorize = reloaded_client.get(
        "/mcp/hormuz/authorize",
        params={
            "response_type": "code",
            "client_id": payload["client_id"],
            "redirect_uri": "https://example.com/callback",
            "scope": payload["scope"],
            "state": "persisted",
        },
    )
    assert authorize.status_code == 200
    assert 'value="approve"' in authorize.text


def test_refresh_token_grant_rotates_token() -> None:
    server = OAuthUiServer(resource_path="/mcp/hormuz")
    client = TestClient(_app_for(server))

    register = client.post(
        "/mcp/hormuz/register",
        json={
            "client_name": "Refresh Client",
            "redirect_uris": ["https://example.com/callback"],
            "token_endpoint_auth_method": "none",
        },
    )
    assert register.status_code == 200
    payload = register.json()

    authorize = client.get(
        "/mcp/hormuz/authorize",
        params={
            "response_type": "code",
            "client_id": payload["client_id"],
            "redirect_uri": "https://example.com/callback",
            "scope": payload["scope"],
            "state": "refresh-state",
            "code_challenge": "refresh-verifier",
            "code_challenge_method": "plain",
        },
    )
    request_id = _extract_request_id(authorize.text)

    consent = client.post(
        "/mcp/hormuz/consent",
        data={"request_id": request_id, "decision": "approve"},
        follow_redirects=False,
    )
    query = parse_qs(urlparse(consent.headers["location"]).query)
    code = query["code"][0]

    token = client.post(
        "/mcp/hormuz/token",
        data={
            "grant_type": "authorization_code",
            "client_id": payload["client_id"],
            "redirect_uri": "https://example.com/callback",
            "code": code,
            "code_verifier": "refresh-verifier",
        },
    )
    assert token.status_code == 200
    issued = token.json()
    original_refresh_token = issued["refresh_token"]

    refreshed = client.post(
        "/mcp/hormuz/token",
        data={
            "grant_type": "refresh_token",
            "client_id": payload["client_id"],
            "refresh_token": original_refresh_token,
        },
    )
    assert refreshed.status_code == 200
    refreshed_payload = refreshed.json()
    assert refreshed_payload["refresh_token"] != original_refresh_token
    assert "access_token" in refreshed_payload

    replay = client.post(
        "/mcp/hormuz/token",
        data={
            "grant_type": "refresh_token",
            "client_id": payload["client_id"],
            "refresh_token": original_refresh_token,
        },
    )
    assert replay.status_code == 400
    assert replay.json()["error"] == "invalid_grant"
