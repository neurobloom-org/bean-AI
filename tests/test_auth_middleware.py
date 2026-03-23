"""Tests for api/middleware/auth_middleware.py

Covers:
  - _is_public_path (exact match and prefix match)
  - _normalize_path
  - extract_token_from_request (Bearer header)
  - extract_token_from_websocket (Bearer header, protocol header, query param)
  - _decode_with_hs256 (valid, expired, invalid)
  - decode_supabase_jwt (empty token, unsupported alg)

No real Supabase JWT or JWKS calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import jwt as pyjwt
import pytest

from api.middleware.auth_middleware import (
    PUBLIC_PATHS,
    PUBLIC_PATH_PREFIXES,
    _is_public_path,
    _normalize_path,
    decode_supabase_jwt,
    extract_token_from_request,
    extract_token_from_websocket,
)
from shared.config import reset_settings
from shared.exceptions import AuthError, TokenExpiredError


@pytest.fixture(autouse=True)
def reset_cfg():
    reset_settings()
    yield
    reset_settings()


# ─────────────────────────────────────────────────────────────────────────────
# _normalize_path
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizePath:
    def test_trailing_slash_stripped(self):
        assert _normalize_path("/docs/") == "/docs"

    def test_root_stays_root(self):
        assert _normalize_path("/") == "/"

    def test_empty_string_becomes_root(self):
        assert _normalize_path("") == "/"


# ─────────────────────────────────────────────────────────────────────────────
# _is_public_path
# ─────────────────────────────────────────────────────────────────────────────


class TestIsPublicPath:
    def test_health_is_public(self):
        assert _is_public_path("/api/v1/health") is True

    def test_login_is_public(self):
        assert _is_public_path("/api/v1/auth/login") is True

    def test_internal_prefix_is_public(self):
        assert _is_public_path("/internal/purge") is True
        assert _is_public_path("/internal/health") is True

    def test_protected_route_is_not_public(self):
        assert _is_public_path("/api/v1/sessions") is False
        assert _is_public_path("/api/v1/alerts") is False

    def test_all_declared_public_paths_pass(self):
        for path in PUBLIC_PATHS:
            assert _is_public_path(path) is True

    def test_all_declared_prefixes_pass(self):
        for prefix in PUBLIC_PATH_PREFIXES:
            assert _is_public_path(f"{prefix}some-endpoint") is True


# ─────────────────────────────────────────────────────────────────────────────
# extract_token_from_request
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractTokenFromRequest:
    def test_extracts_bearer_token(self):
        req = MagicMock()
        req.headers = {"Authorization": "Bearer mytoken123"}
        assert extract_token_from_request(req) == "mytoken123"

    def test_returns_none_if_no_header(self):
        req = MagicMock()
        req.headers = {}
        assert extract_token_from_request(req) is None

    def test_returns_none_if_not_bearer(self):
        req = MagicMock()
        req.headers = {"Authorization": "Basic abc123"}
        assert extract_token_from_request(req) is None

    def test_strips_whitespace_from_token(self):
        req = MagicMock()
        req.headers = {"Authorization": "Bearer   mytoken   "}
        assert extract_token_from_request(req) == "mytoken"


# ─────────────────────────────────────────────────────────────────────────────
# extract_token_from_websocket
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractTokenFromWebSocket:
    def test_extracts_from_bearer_protocol(self):
        ws = MagicMock()
        ws.headers = {"Sec-WebSocket-Protocol": "bearer.mytoken123", "Authorization": ""}
        ws.query_params = {}
        result = extract_token_from_websocket(ws)
        assert result == "mytoken123"

    def test_extracts_from_authorization_header(self):
        ws = MagicMock()
        ws.headers = {"Sec-WebSocket-Protocol": "", "Authorization": "Bearer wstoken"}
        ws.query_params = {}
        result = extract_token_from_websocket(ws)
        assert result == "wstoken"

    def test_extracts_from_query_param(self):
        ws = MagicMock()
        ws.headers = {"Sec-WebSocket-Protocol": "", "Authorization": ""}
        ws.query_params = {"token": "querytoken"}
        result = extract_token_from_websocket(ws)
        assert result == "querytoken"

    def test_returns_none_if_no_token(self):
        ws = MagicMock()
        ws.headers = {"Sec-WebSocket-Protocol": "", "Authorization": ""}
        ws.query_params = {}
        result = extract_token_from_websocket(ws)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# decode_supabase_jwt — top-level dispatch
# ─────────────────────────────────────────────────────────────────────────────


class TestDecodeSupabaseJwt:
    @pytest.mark.asyncio
    async def test_empty_token_raises_auth_error(self):
        with pytest.raises(AuthError, match="empty"):
            await decode_supabase_jwt("")

    @pytest.mark.asyncio
    async def test_whitespace_token_raises_auth_error(self):
        with pytest.raises(AuthError, match="empty"):
            await decode_supabase_jwt("   ")

    @pytest.mark.asyncio
    async def test_unsupported_algorithm_raises_auth_error(self):
        # Build a token with HS512 (not supported)
        token = pyjwt.encode(
            {"sub": "user1", "exp": 9999999999, "aud": "authenticated", "iss": "test"},
            "secret",
            algorithm="HS512",
        )
        with pytest.raises(AuthError, match="Unsupported"):
            await decode_supabase_jwt(token)

    @pytest.mark.asyncio
    async def test_malformed_token_raises_auth_error(self):
        with pytest.raises(AuthError):
            await decode_supabase_jwt("not.a.jwt.at.all")

    @pytest.mark.asyncio
    async def test_hs256_expired_token_raises_token_expired_error(self):
        import time
        token = pyjwt.encode(
            {
                "sub": "user1",
                "aud": "authenticated",
                "iss": "https://dummy.supabase.co/auth/v1",
                "exp": int(time.time()) - 3600,  # expired 1h ago
            },
            "dummy-jwt-secret",
            algorithm="HS256",
        )

        with patch("api.middleware.auth_middleware.get_settings") as mock_settings:
            mock_settings.return_value.supabase_url = "https://dummy.supabase.co"
            mock_settings.return_value.supabase_jwt_secret = "dummy-jwt-secret"
            with pytest.raises(TokenExpiredError):
                await decode_supabase_jwt(token)

    @pytest.mark.asyncio
    async def test_hs256_valid_token_returns_payload(self):
        import time
        token = pyjwt.encode(
            {
                "sub": "user-abc",
                "aud": "authenticated",
                "iss": "https://dummy.supabase.co/auth/v1",
                "exp": int(time.time()) + 3600,
            },
            "dummy-jwt-secret",
            algorithm="HS256",
        )

        with patch("api.middleware.auth_middleware.get_settings") as mock_settings:
            mock_settings.return_value.supabase_url = "https://dummy.supabase.co"
            mock_settings.return_value.supabase_jwt_secret = "dummy-jwt-secret"
            payload = await decode_supabase_jwt(token)

        assert payload["sub"] == "user-abc"

    @pytest.mark.asyncio
    async def test_hs256_no_secret_configured_raises_auth_error(self):
        import time
        token = pyjwt.encode(
            {
                "sub": "user1",
                "aud": "authenticated",
                "iss": "https://dummy.supabase.co/auth/v1",
                "exp": int(time.time()) + 3600,
            },
            "any-secret",
            algorithm="HS256",
        )

        with patch("api.middleware.auth_middleware.get_settings") as mock_settings:
            mock_settings.return_value.supabase_url = "https://dummy.supabase.co"
            mock_settings.return_value.supabase_jwt_secret = ""
            with pytest.raises(AuthError, match="SUPABASE_JWT_SECRET"):
                await decode_supabase_jwt(token)
