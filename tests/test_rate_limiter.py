"""Tests for api/middleware/rate_limiter.py

Covers:
  - _normalize_path
  - _hash_key (salt presence, determinism, truncation)
  - _get_request_identifier (user vs IP)
  - _add_rate_limit_headers
  - _can_mutate_response_headers
  - check_rate_limit (success, RPC failure → deny)
  - check_ws_rate_limit
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.responses import Response, StreamingResponse

from api.middleware.rate_limiter import (
    _add_rate_limit_headers,
    _can_mutate_response_headers,
    _get_request_identifier,
    _hash_key,
    _normalize_path,
    check_rate_limit,
    check_ws_rate_limit,
)
from shared.config import reset_settings


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

    def test_root_preserved(self):
        assert _normalize_path("/") == "/"

    def test_no_trailing_slash_unchanged(self):
        assert _normalize_path("/api/v1/health") == "/api/v1/health"

    def test_multiple_trailing_slashes(self):
        assert _normalize_path("/docs///") == "/docs"


# ─────────────────────────────────────────────────────────────────────────────
# _hash_key
# ─────────────────────────────────────────────────────────────────────────────


class TestHashKey:
    def test_deterministic(self):
        with patch("api.middleware.rate_limiter.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_hash_salt = "test-salt"
            h1 = _hash_key("user:abc")
            h2 = _hash_key("user:abc")
        assert h1 == h2

    def test_different_keys_produce_different_hashes(self):
        with patch("api.middleware.rate_limiter.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_hash_salt = "test-salt"
            h1 = _hash_key("user:abc")
            h2 = _hash_key("user:xyz")
        assert h1 != h2

    def test_output_is_32_chars(self):
        with patch("api.middleware.rate_limiter.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_hash_salt = "test-salt"
            result = _hash_key("user:abc")
        assert len(result) == 32

    def test_no_salt_still_works(self):
        with patch("api.middleware.rate_limiter.get_settings") as mock_settings:
            mock_settings.return_value.rate_limit_hash_salt = ""
            result = _hash_key("user:abc")
        assert isinstance(result, str)
        assert len(result) == 32


# ─────────────────────────────────────────────────────────────────────────────
# _get_request_identifier
# ─────────────────────────────────────────────────────────────────────────────


class TestGetRequestIdentifier:
    def test_prefers_user_id_from_state(self):
        req = MagicMock()
        req.state.user_id = "user-abc"
        result = _get_request_identifier(req)
        assert result == "user:user-abc"

    def test_falls_back_to_ip(self):
        req = MagicMock(spec=["state", "client"])
        req.state = MagicMock(spec=[])  # no user_id attribute
        req.client = MagicMock()
        req.client.host = "192.168.1.1"
        result = _get_request_identifier(req)
        assert result == "ip:192.168.1.1"

    def test_no_client_uses_unknown(self):
        req = MagicMock(spec=["state", "client"])
        req.state = MagicMock(spec=[])
        req.client = None
        result = _get_request_identifier(req)
        assert result == "ip:unknown"


# ─────────────────────────────────────────────────────────────────────────────
# _add_rate_limit_headers
# ─────────────────────────────────────────────────────────────────────────────


class TestAddRateLimitHeaders:
    def test_headers_set_correctly(self):
        response = Response()
        _add_rate_limit_headers(response, limit=100, remaining=42, window_seconds=60)
        assert response.headers["X-RateLimit-Limit"] == "100"
        assert response.headers["X-RateLimit-Remaining"] == "42"
        assert response.headers["X-RateLimit-Window-Seconds"] == "60"

    def test_remaining_clamped_to_zero(self):
        response = Response()
        _add_rate_limit_headers(response, limit=10, remaining=-5, window_seconds=60)
        assert response.headers["X-RateLimit-Remaining"] == "0"


# ─────────────────────────────────────────────────────────────────────────────
# _can_mutate_response_headers
# ─────────────────────────────────────────────────────────────────────────────


class TestCanMutateResponseHeaders:
    def test_regular_response_returns_true(self):
        assert _can_mutate_response_headers(Response()) is True

    def test_streaming_response_returns_false(self):
        assert _can_mutate_response_headers(StreamingResponse(iter([]))) is False


# ─────────────────────────────────────────────────────────────────────────────
# check_rate_limit
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckRateLimit:
    @pytest.mark.asyncio
    async def test_allowed_when_under_limit(self):
        mock_client = AsyncMock()
        mock_client.rpc.return_value.execute = AsyncMock(return_value=MagicMock(data=5))

        with patch("api.middleware.rate_limiter.get_service_client", return_value=mock_client):
            with patch("api.middleware.rate_limiter.get_settings") as mock_settings:
                mock_settings.return_value.rate_limit_hash_salt = "salt"
                allowed, count = await check_rate_limit("user:abc", max_requests=100)

        assert allowed is True
        assert count == 5

    @pytest.mark.asyncio
    async def test_denied_when_over_limit(self):
        mock_client = AsyncMock()
        mock_client.rpc.return_value.execute = AsyncMock(return_value=MagicMock(data=101))

        with patch("api.middleware.rate_limiter.get_service_client", return_value=mock_client):
            with patch("api.middleware.rate_limiter.get_settings") as mock_settings:
                mock_settings.return_value.rate_limit_hash_salt = "salt"
                allowed, count = await check_rate_limit("user:abc", max_requests=100)

        assert allowed is False
        assert count == 101

    @pytest.mark.asyncio
    async def test_rpc_failure_denies_for_safety(self):
        mock_client = AsyncMock()
        mock_client.rpc.return_value.execute = AsyncMock(side_effect=RuntimeError("DB down"))

        with patch("api.middleware.rate_limiter.get_service_client", return_value=mock_client):
            with patch("api.middleware.rate_limiter.get_settings") as mock_settings:
                mock_settings.return_value.rate_limit_hash_salt = "salt"
                allowed, count = await check_rate_limit("user:abc", max_requests=100)

        assert allowed is False
        assert count == 0


# ─────────────────────────────────────────────────────────────────────────────
# check_ws_rate_limit
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckWsRateLimit:
    @pytest.mark.asyncio
    async def test_allowed_returns_true(self):
        with patch("api.middleware.rate_limiter.check_rate_limit", new_callable=AsyncMock, return_value=(True, 10)):
            with patch("api.middleware.rate_limiter.get_settings") as mock_settings:
                mock_settings.return_value.rate_limit_ws_messages_per_min = 60
                result = await check_ws_rate_limit("user-abc")
        assert result is True

    @pytest.mark.asyncio
    async def test_denied_returns_false(self):
        with patch("api.middleware.rate_limiter.check_rate_limit", new_callable=AsyncMock, return_value=(False, 61)):
            with patch("api.middleware.rate_limiter.get_settings") as mock_settings:
                mock_settings.return_value.rate_limit_ws_messages_per_min = 60
                result = await check_ws_rate_limit("user-abc")
        assert result is False
