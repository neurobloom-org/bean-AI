"""BEAN AI — Supabase-backed rate limiter.

All counter operations are routed through the Postgres RPC function
`check_and_increment_rate_limit` (migration 009), which performs an atomic
INSERT … ON CONFLICT DO UPDATE in a single statement. This eliminates the
read-then-update race condition that existed in the previous implementation.

Public API:
    RateLimiterMiddleware   — Starlette HTTP middleware; applies per-user/IP limits.
    check_ws_rate_limit()   — standalone coroutine for WebSocket message gating.

Internal:
    check_rate_limit()      — shared atomic check+increment helper.
    _hash_key()             — salted SHA-256 of raw identifiers before DB storage.

Rate-limit counter cleanup:
    Stale rows are deleted by services/cleanup_service.clean_expired_rate_limits(),
    called from background/session_cleanup.run_transcript_purge_loop().
    This module does NOT own the cleanup loop.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from typing import Final

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from services.supabase_client import get_service_client
from shared.config import get_settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

PUBLIC_RATE_LIMIT_EXEMPT_PATHS: Final[frozenset[str]] = frozenset(
    {
        "/",
        "/health",
        "/api/v1/health",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)

_WINDOW_SECONDS: Final[int] = 60


# ── Key helpers ───────────────────────────────────────────────────────────────


def _normalize_path(path: str) -> str:
    normalized = path.rstrip("/")
    return normalized or "/"


def _hash_key(raw_key: str) -> str:
    settings = get_settings()
    salt = getattr(settings, "rate_limit_hash_salt", None)

    if not salt:
        logger.warning(
            "RATE_LIMIT_HASH_SALT is not configured; falling back to unsalted hashing"
        )
        material = raw_key
    else:
        material = f"{salt}:{raw_key}"

    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _get_request_identifier(request: Request) -> str:
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"

    client_ip = request.client.host if request.client else "unknown"
    return f"ip:{client_ip}"


def _add_rate_limit_headers(
    response: Response,
    *,
    limit: int,
    remaining: int,
    window_seconds: int,
) -> None:
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
    response.headers["X-RateLimit-Window-Seconds"] = str(window_seconds)


def _can_mutate_response_headers(response: Response) -> bool:
    return not isinstance(response, StreamingResponse)


# ── Core atomic helper ────────────────────────────────────────────────────────


async def check_rate_limit(
    key: str,
    max_requests: int,
    window_seconds: int = _WINDOW_SECONDS,
) -> tuple[bool, int]:
    hashed_key = _hash_key(key)

    try:
        client = await get_service_client()
        result = await client.rpc(
            "check_and_increment_rate_limit",
            {"p_key": hashed_key, "p_window_seconds": window_seconds},
        ).execute()

        # ✅ FIX: handle list response like [5]
        data = result.data
        if isinstance(data, list):
            count = int(data[0]) if data else 0
        else:
            count = int(data)

        return count <= max_requests, count

    except Exception as exc:
        logger.error(
            "Rate limit RPC failed for key=%s: %s — denying request for safety",
            f"{hashed_key[:8]}...",
            exc,
        )
        return False, 0


# ── HTTP Middleware ───────────────────────────────────────────────────────────


class RateLimiterMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        settings = get_settings()
        path = _normalize_path(request.url.path)

        if request.method.upper() == "OPTIONS":
            return await call_next(request)

        if path in PUBLIC_RATE_LIMIT_EXEMPT_PATHS:
            return await call_next(request)

        if path.startswith("/internal/"):
            return await call_next(request)

        identifier = _get_request_identifier(request)
        limit = settings.rate_limit_api_calls_per_min

        allowed, count = await check_rate_limit(
            key=f"api:{identifier}",
            max_requests=limit,
            window_seconds=_WINDOW_SECONDS,
        )

        if not allowed:
            logger.warning(
                "Rate limit exceeded [identifier=%s count=%d limit=%d path=%s]",
                identifier,
                count,
                limit,
                path,
            )
            rate_limit_response = JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests. Please slow down.",
                    "retry_after_seconds": _WINDOW_SECONDS,
                },
                headers={"Retry-After": str(_WINDOW_SECONDS)},
            )
            _add_rate_limit_headers(
                rate_limit_response,
                limit=limit,
                remaining=0,
                window_seconds=_WINDOW_SECONDS,
            )
            return rate_limit_response

        response = await call_next(request)

        if _can_mutate_response_headers(response):
            _add_rate_limit_headers(
                response,
                limit=limit,
                remaining=limit - count,
                window_seconds=_WINDOW_SECONDS,
            )

        return response


# ── WebSocket helper ──────────────────────────────────────────────────────────


async def check_ws_rate_limit(user_id: str) -> bool:
    settings = get_settings()
    limit = settings.rate_limit_ws_messages_per_min

    allowed, count = await check_rate_limit(
        key=f"ws:{user_id}",
        max_requests=limit,
        window_seconds=_WINDOW_SECONDS,
    )

    if not allowed:
        logger.warning(
            "WS rate limit hit [user=%s count=%d limit=%d]",
            user_id,
            count,
            limit,
        )

    return allowed