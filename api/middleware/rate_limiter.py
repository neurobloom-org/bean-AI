"""BEAN AI — Supabase-backed rate limiter.

Notes:
- This implementation is shared across replicas because it stores counters
  in Supabase instead of local memory.
- IMPORTANT: The current read-then-update pattern is NOT fully atomic under
  concurrent load. For strict production-grade enforcement across replicas,
  move the check/increment logic into a Postgres RPC function.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Final

from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from services.supabase_client import get_service_client
from shared.config import get_settings

logger = logging.getLogger(__name__)

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


def _normalize_path(path: str) -> str:
    """Normalize paths so /docs and /docs/ are treated the same."""
    normalized = path.rstrip("/")
    return normalized or "/"


def _hash_key(raw_key: str) -> str:
    """Hash the rate-limit key with a secret salt.

    This avoids storing plaintext IPs/user IDs and makes precomputed hash
    guessing much harder than a plain unsalted SHA-256.
    """
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


def _parse_iso_datetime(value: str) -> datetime:
    """Parse an ISO timestamp safely."""
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _get_request_identifier(request: Request) -> str:
    """Return a stable identifier for rate limiting.

    Prefer authenticated user_id when available.
    Fall back to client IP if unauthenticated.
    """
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
    """Attach helpful rate-limit headers to the response."""
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
    response.headers["X-RateLimit-Window-Seconds"] = str(window_seconds)


def _can_mutate_response_headers(response: Response) -> bool:
    """Return whether it is safe to add headers after call_next().

    For streaming responses, headers may already be committed by the time
    middleware regains control, so mutating them can be unsafe.
    """
    return not isinstance(response, StreamingResponse)


async def check_rate_limit(
    key: str,
    max_requests: int,
    window_minutes: int = 1,
) -> tuple[bool, int]:
    """Check and increment a rate-limit counter in Supabase.

    Args:
        key: Raw identifier (user_id, IP, etc.). This will be hashed before
             storage so raw identifiers are not persisted in the DB.
        max_requests: Maximum allowed requests during the window.
        window_minutes: Window duration in minutes.

    Returns:
        tuple[allowed, current_count]
        - allowed=True means the request is permitted
        - allowed=False means the caller should be rate-limited

    Important:
        This implementation is not fully atomic under concurrent load because
        it uses a read-then-update pattern. For strict correctness across
        replicas, move this into a database RPC / transactional SQL function.
    """
    hashed_key = _hash_key(key)
    now = datetime.now(UTC)
    window_start_threshold = now - timedelta(minutes=window_minutes)

    try:
        client = await get_service_client()

        result = (
            await client.table("rate_limits")
            .select("id, request_count, window_start")
            .eq("key", hashed_key)
            .maybe_single()
            .execute()
        )

        if result.data:
            record = result.data
            record_window_start = _parse_iso_datetime(record["window_start"])

            if record_window_start < window_start_threshold:
                await (
                    client.table("rate_limits")
                    .update(
                        {
                            "request_count": 1,
                            "window_start": now.isoformat(),
                        }
                    )
                    .eq("key", hashed_key)
                    .execute()
                )
                return True, 1

            current = int(record["request_count"])
            if current >= max_requests:
                return False, current

            next_count = current + 1
            await (
                client.table("rate_limits")
                .update({"request_count": next_count})
                .eq("key", hashed_key)
                .execute()
            )
            return True, next_count

        await (
            client.table("rate_limits")
            .insert(
                {
                    "key": hashed_key,
                    "request_count": 1,
                    "window_start": now.isoformat(),
                }
            )
            .execute()
        )
        return True, 1

    except Exception as exc:
        logger.error(
            "Rate limit check failed for key=%s: %s — failing open",
            f"{hashed_key[:8]}...",
            exc,
        )
        return True, 0


async def clean_expired_rate_limits() -> int:
    """Remove stale rate-limit records older than 1 hour."""
    try:
        threshold = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
        client = await get_service_client()
        result = (
            await client.table("rate_limits")
            .delete()
            .lt("window_start", threshold)
            .execute()
        )

        count = len(result.data) if result.data else 0
        if count:
            logger.debug("Cleaned %d stale rate limit records", count)
        return count
    except Exception as exc:
        logger.error("Rate limit cleanup failed: %s", exc)
        return 0


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """HTTP API rate-limiting middleware backed by Supabase."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        settings = get_settings()
        path = _normalize_path(request.url.path)

        if request.method.upper() == "OPTIONS":
            return await call_next(request)

        if path in PUBLIC_RATE_LIMIT_EXEMPT_PATHS:
            return await call_next(request)

        identifier = _get_request_identifier(request)
        key = f"api:{identifier}"
        limit = settings.rate_limit_api_calls_per_min
        window_seconds = 60

        allowed, count = await check_rate_limit(
            key=key,
            max_requests=limit,
            window_minutes=1,
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
                    "retry_after_seconds": window_seconds,
                },
                headers={"Retry-After": str(window_seconds)},
            )
            _add_rate_limit_headers(
                rate_limit_response,
                limit=limit,
                remaining=0,
                window_seconds=window_seconds,
            )
            return rate_limit_response

        response = await call_next(request)

        if _can_mutate_response_headers(response):
            remaining = limit - count
            _add_rate_limit_headers(
                response,
                limit=limit,
                remaining=remaining,
                window_seconds=window_seconds,
            )

        return response


async def check_ws_rate_limit(user_id: str) -> bool:
    """Check WebSocket message rate limit for a user.

    Returns True if allowed, False if rate-limited.
    Intended to be called per WebSocket message or audio frame.
    """
    settings = get_settings()
    limit = settings.rate_limit_ws_messages_per_min

    allowed, count = await check_rate_limit(
        key=f"ws:{user_id}",
        max_requests=limit,
        window_minutes=1,
    )

    if not allowed:
        logger.warning(
            "WS rate limit hit [user=%s count=%d limit=%d]",
            user_id,
            count,
            limit,
        )

    return allowed