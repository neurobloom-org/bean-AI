"""BEAN AI — Supabase JWT authentication middleware.

Updated for modern Supabase Auth.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from typing import Any, cast

import httpx
import jwt as pyjwt
from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse
from jwt import algorithms
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from shared.config import get_settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Public routes
# ─────────────────────────────────────────────────────────────────────────────

PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/health",
        "/api/v1/health",
        "/api/v1/auth/login",
        "/api/v1/auth/signup",
        "/api/v1/auth/callback",
        "/api/v1/auth/refresh",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)

PUBLIC_PATH_PREFIXES: tuple[str, ...] = ()

# ─────────────────────────────────────────────────────────────────────────────
# JWKS cache
# ─────────────────────────────────────────────────────────────────────────────

_JWKS_CACHE: dict[str, dict[str, Any]] = {}
_JWKS_CACHE_TTL_SECONDS = 300
_JWKS_LOCK = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _expected_issuer() -> str:
    """Return the expected JWT issuer for this Supabase project."""
    settings = get_settings()
    return f"{settings.supabase_url.rstrip('/')}/auth/v1"


def _jwks_url() -> str:
    """Return the JWKS endpoint for this Supabase project."""
    settings = get_settings()
    return f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"


def _normalize_path(path: str) -> str:
    """Normalize request paths so /docs and /docs/ are treated the same."""
    normalized = path.rstrip("/")
    return normalized or "/"


def _is_public_path(path: str) -> bool:
    """Return True if the path is public."""
    if path in PUBLIC_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in PUBLIC_PATH_PREFIXES)


async def _fetch_jwks(force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Fetch and cache the JWKS document from Supabase."""
    url = _jwks_url()
    now = time.time()

    cached = _JWKS_CACHE.get(url)
    if not force_refresh and cached and cached["expires_at"] > now:
        return cast(dict[str, dict[str, Any]], cached["data"])

    async with _JWKS_LOCK:
        cached = _JWKS_CACHE.get(url)
        if not force_refresh and cached and cached["expires_at"] > now:
            return cast(dict[str, dict[str, Any]], cached["data"])

        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()

        raw_keys = data.get("keys")
        if not isinstance(raw_keys, list):
            raise ValueError("Invalid JWKS response from Supabase: missing keys list")

        jwks_by_kid: dict[str, dict[str, Any]] = {}
        for key in raw_keys:
            if isinstance(key, dict) and key.get("kid"):
                jwks_by_kid[key["kid"]] = key

        if not jwks_by_kid:
            raise ValueError(
                "Invalid JWKS response from Supabase: no usable keys found"
            )

        _JWKS_CACHE[url] = {
            "data": jwks_by_kid,
            "expires_at": now + _JWKS_CACHE_TTL_SECONDS,
        }
        return jwks_by_kid


def _get_token_header(token: str) -> dict[str, Any]:
    """Read the unverified JWT header."""
    try:
        return cast(dict[str, Any], pyjwt.get_unverified_header(token))
    except pyjwt.InvalidTokenError as exc:
        raise ValueError(f"Invalid token header: {exc}") from exc


def _decode_with_hs256(token: str) -> dict[str, Any]:
    """Validate legacy HS256 Supabase JWTs."""
    settings = get_settings()

    if not settings.supabase_jwt_secret:
        raise ValueError(
            "HS256 token received but SUPABASE_JWT_SECRET is not configured"
        )

    try:
        # FIX Line 128: Hard cast the result of decode directly
        return cast(
            dict[str, Any],
            pyjwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
                issuer=_expected_issuer(),
                options={"require": ["exp", "sub", "aud", "iss"]},
            ),
        )
    except pyjwt.ExpiredSignatureError as exc:
        raise ValueError("Token expired — please re-authenticate") from exc
    except pyjwt.InvalidTokenError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc


async def _decode_with_jwks(token: str, kid: str) -> dict[str, Any]:
    """Validate RS256 Supabase JWTs using the project's JWKS."""
    try:
        jwks_by_kid = await _fetch_jwks()
    except Exception as exc:
        raise ValueError(f"Failed to fetch Supabase JWKS: {exc}") from exc

    matching_key = jwks_by_kid.get(kid)
    if not matching_key:
        jwks_by_kid = await _fetch_jwks(force_refresh=True)
        matching_key = jwks_by_kid.get(kid)

    if not matching_key:
        raise ValueError("No matching signing key found for token")

    try:
        public_key = cast(RSAPublicKey, algorithms.RSAAlgorithm.from_jwk(matching_key))
        # FIX: Hard cast here as well
        return cast(
            dict[str, Any],
            pyjwt.decode(
                token,
                public_key,
                algorithms=["RS256"],
                audience="authenticated",
                issuer=_expected_issuer(),
                options={"require": ["exp", "sub", "aud", "iss"]},
            ),
        )
    except pyjwt.ExpiredSignatureError as exc:
        raise ValueError("Token expired — please re-authenticate") from exc
    except pyjwt.InvalidTokenError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc


async def decode_supabase_jwt(token: str) -> dict[str, Any]:
    """Decode and validate a Supabase-issued JWT."""
    header = _get_token_header(token)
    alg = header.get("alg")
    kid = header.get("kid")

    if alg == "HS256":
        return _decode_with_hs256(token)

    if alg == "RS256":
        if not kid:
            raise ValueError("Missing key ID (kid) in token header")
        return await _decode_with_jwks(token, str(kid))

    raise ValueError(f"Unsupported JWT algorithm: {alg}")


# ─────────────────────────────────────────────────────────────────────────────
# Token extraction helpers
# ─────────────────────────────────────────────────────────────────────────────


def extract_token_from_request(request: Request) -> str | None:
    """Extract JWT from Authorization header only."""
    auth_header = str(request.headers.get("Authorization", ""))
    if auth_header.startswith("Bearer "):
        return str(auth_header[7:].strip())
    return None


def extract_token_from_websocket(websocket: WebSocket) -> str | None:
    """Extract JWT from a WebSocket connection."""
    protocol_header = str(websocket.headers.get("Sec-WebSocket-Protocol", ""))
    for proto in protocol_header.split(","):
        proto = proto.strip()
        if proto.startswith("bearer."):
            return str(proto[len("bearer.") :])

    auth_header = str(websocket.headers.get("Authorization", ""))
    if auth_header.startswith("Bearer "):
        return str(auth_header[7:].strip())

    return None


# ─────────────────────────────────────────────────────────────────────────────
# HTTP middleware
# ─────────────────────────────────────────────────────────────────────────────


class SupabaseAuthMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """Validate Supabase JWT on every non-public HTTP request."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = _normalize_path(request.url.path)

        if request.method.upper() == "OPTIONS":
            return await call_next(request)

        if _is_public_path(path):
            return await call_next(request)

        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await call_next(request)

        token = extract_token_from_request(request)
        if not token:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Authorization header required. Use: Authorization: Bearer <token>"
                },
            )

        try:
            payload = await decode_supabase_jwt(token)

            request.state.user_id = payload["sub"]
            request.state.email = payload.get("email")
            request.state.role = payload.get("role")
            request.state.jwt_payload = payload
        except ValueError as exc:
            logger.warning("Auth failed [path=%s]: %s", path, exc)
            return JSONResponse(
                status_code=401,
                content={"detail": str(exc)},
            )

        return await call_next(request)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket auth helper
# ─────────────────────────────────────────────────────────────────────────────


async def authenticate_websocket(websocket: WebSocket) -> tuple[str, dict[str, Any]]:
    """Authenticate a WebSocket connection and return (user_id, jwt_payload)."""
    token = extract_token_from_websocket(websocket)
    if not token:
        await websocket.close(
            code=4401,
            reason="Missing token. Use Sec-WebSocket-Protocol: bearer.<token>",
        )
        raise WebSocketAuthError("No token provided")

    try:
        payload = await decode_supabase_jwt(token)
        user_id = str(payload["sub"])
        return user_id, payload
    except ValueError as exc:
        await websocket.close(code=4401, reason=str(exc))
        raise WebSocketAuthError(str(exc)) from exc


class WebSocketAuthError(Exception):
    """Raised when WebSocket authentication fails."""
