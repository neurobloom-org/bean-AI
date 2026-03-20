"""BEAN AI — Supabase JWT authentication middleware.

Updated for modern Supabase Auth.

What this version improves:
  ✓ Supports BOTH legacy HS256 tokens and newer asymmetric Supabase JWTs
  ✓ Validates JWT audience + issuer
  ✓ Requires critical claims like exp, sub, aud, iss
  ✓ Never accepts JWTs from query parameters
  ✓ Keeps WebSocket auth via Sec-WebSocket-Protocol: bearer.<token>
  ✓ Uses in-memory JWKS caching to avoid fetching keys on every request
  ✓ Normalizes public paths so /docs and /docs/ both work
  ✓ Retries JWKS fetch once if signing keys rotate
  ✓ Exposes role claim on request.state for downstream authorization logic
  ✓ Allows CORS preflight OPTIONS requests through auth middleware
  ✓ Supports exact public paths and optional public path prefixes
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
# EXACT public paths are fast O(1) lookups.
# PUBLIC_PATH_PREFIXES are optional and useful for dynamic public endpoints
# like /api/v1/public/profiles/{user_id}.
# Keep prefixes minimal and intentional to avoid over-exposing routes.

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

PUBLIC_PATH_PREFIXES: tuple[str, ...] = (
    # Example:
    # "/api/v1/public/",
)

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
    """Return True if the path is public.

    Supports:
      - exact public paths
      - optional prefix-based public route families
    """
    if path in PUBLIC_PATHS:
        return True

    return any(path.startswith(prefix) for prefix in PUBLIC_PATH_PREFIXES)


async def _fetch_jwks(force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    """Fetch and cache the JWKS document from Supabase.

    Returns a dictionary keyed by `kid` for O(1) lookup.
    """
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
            raise ValueError("Invalid JWKS response from Supabase: no usable keys found")

        _JWKS_CACHE[url] = {
            "data": jwks_by_kid,
            "expires_at": now + _JWKS_CACHE_TTL_SECONDS,
        }
        return jwks_by_kid


def _get_token_header(token: str) -> dict[str, Any]:
    """Read the unverified JWT header so we can choose the validation method."""
    try:
        return pyjwt.get_unverified_header(token)
    except pyjwt.InvalidTokenError as exc:
        raise ValueError(f"Invalid token header: {exc}") from exc


def _decode_with_hs256(token: str) -> dict[str, Any]:
    """Validate legacy HS256 Supabase JWTs using SUPABASE_JWT_SECRET."""
    settings = get_settings()

    if not settings.supabase_jwt_secret:
        raise ValueError(
            "HS256 token received but SUPABASE_JWT_SECRET is not configured"
        )

    try:
        payload = pyjwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            # Supabase standard end-user JWTs use aud="authenticated".
            audience="authenticated",
            issuer=_expected_issuer(),
            options={
                "require": ["exp", "sub", "aud", "iss"],
            },
        )
        return payload
    except pyjwt.ExpiredSignatureError as exc:
        raise ValueError("Token expired — please re-authenticate") from exc
    except pyjwt.InvalidAudienceError as exc:
        raise ValueError("Invalid token audience") from exc
    except pyjwt.InvalidIssuerError as exc:
        raise ValueError("Invalid token issuer") from exc
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
        try:
            jwks_by_kid = await _fetch_jwks(force_refresh=True)
        except Exception as exc:
            raise ValueError(f"Failed to refresh Supabase JWKS: {exc}") from exc
        matching_key = jwks_by_kid.get(kid)

    if not matching_key:
        raise ValueError("No matching signing key found for token")

    try:
        public_key = cast(RSAPublicKey, algorithms.RSAAlgorithm.from_jwk(matching_key))
        payload = pyjwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            # Supabase standard end-user JWTs use aud="authenticated".
            audience="authenticated",
            issuer=_expected_issuer(),
            options={
                "require": ["exp", "sub", "aud", "iss"],
            },
        )
        return payload
    except pyjwt.ExpiredSignatureError as exc:
        raise ValueError("Token expired — please re-authenticate") from exc
    except pyjwt.InvalidAudienceError as exc:
        raise ValueError("Invalid token audience") from exc
    except pyjwt.InvalidIssuerError as exc:
        raise ValueError("Invalid token issuer") from exc
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
        return await _decode_with_jwks(token, kid)

    raise ValueError(f"Unsupported JWT algorithm: {alg}")


# ─────────────────────────────────────────────────────────────────────────────
# Token extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_token_from_request(request: Request) -> str | None:
    """Extract JWT from Authorization header only."""
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return None


def extract_token_from_websocket(websocket: WebSocket) -> str | None:
    """Extract JWT from a WebSocket connection.

    Priority:
      1. Sec-WebSocket-Protocol: bearer.<token>
      2. Authorization: Bearer <token>
    """
    protocol_header = websocket.headers.get("Sec-WebSocket-Protocol", "")
    for proto in protocol_header.split(","):
        proto = proto.strip()
        if proto.startswith("bearer."):
            return proto[len("bearer."):]

    auth_header = websocket.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    return None


# ─────────────────────────────────────────────────────────────────────────────
# HTTP middleware
# ─────────────────────────────────────────────────────────────────────────────

class SupabaseAuthMiddleware(BaseHTTPMiddleware):
    """Validate Supabase JWT on every non-public HTTP request."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        path = _normalize_path(request.url.path)

        # Let CORS preflight requests pass through.
        # This avoids rejecting browser OPTIONS requests that do not include
        # Authorization headers.
        if request.method.upper() == "OPTIONS":
            return await call_next(request)

        if _is_public_path(path):
            return await call_next(request)

        # WebSocket upgrades are authenticated inside the WS route itself.
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await call_next(request)

        token = extract_token_from_request(request)
        if not token:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": "Authorization header required. "
                    "Use: Authorization: Bearer <token>"
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
    """Authenticate a WebSocket connection and return (user_id, jwt_payload).

    Note:
    - This helper validates the token but does NOT call websocket.accept().
    - Your route should accept the socket after successful auth.
    - If the client sent Sec-WebSocket-Protocol: bearer.<token>, strict browser
      clients may expect the accepted subprotocol to match.
    """
    token = extract_token_from_websocket(websocket)
    if not token:
        await websocket.close(
            code=4401,
            reason="Missing token. Use Sec-WebSocket-Protocol: bearer.<token>",
        )
        raise WebSocketAuthError("No token provided")

    try:
        payload = await decode_supabase_jwt(token)
        user_id = payload["sub"]
        logger.debug(
            "WebSocket authenticated: user=%s role=%s",
            user_id,
            payload.get("role"),
        )
        return user_id, payload
    except ValueError as exc:
        await websocket.close(code=4401, reason=str(exc))
        raise WebSocketAuthError(str(exc)) from exc


class WebSocketAuthError(Exception):
    """Raised when WebSocket authentication fails."""