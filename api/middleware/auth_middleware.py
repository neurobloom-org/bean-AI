"""BEAN AI — Supabase JWT authentication middleware.

Supports both HS256 (legacy Supabase) and RS256 (modern Supabase) JWTs.

Public API:
    decode_supabase_jwt(token)          — decode + validate any Supabase JWT
    extract_token_from_request(request) — pull Bearer token from HTTP request
    extract_token_from_websocket(ws)    — pull Bearer token from WS handshake
    authenticate_websocket(ws)          — full WS auth, returns (user_id, payload)
    SupabaseAuthMiddleware              — Starlette middleware for HTTP routes

Exception hierarchy (all from shared.exceptions):
    AuthError           — base for all auth failures
    TokenExpiredError   — JWT has expired (subclass of AuthError)
    WebSocketAuthError  — WS-specific auth failure (subclass of AuthError)

Note for callers of decode_supabase_jwt() outside this module
(e.g. api/routes/auth.py get_current_user_id):
    decode_supabase_jwt now raises AuthError / TokenExpiredError instead of
    ValueError. Update catch clauses to: except (AuthError, ValueError)
    during migration, or just except AuthError going forward.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any, cast

import httpx
import jwt as pyjwt
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from fastapi import Request, WebSocket
from fastapi.responses import JSONResponse
from jwt import algorithms
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from shared.config import get_settings
from shared.exceptions import AuthError, TokenExpiredError, WebSocketAuthError

__all__ = [
    "decode_supabase_jwt",
    "extract_token_from_request",
    "extract_token_from_websocket",
    "authenticate_websocket",
    "SupabaseAuthMiddleware",
    "PUBLIC_PATHS",
    "PUBLIC_PATH_PREFIXES",
]

logger = logging.getLogger(__name__)

# ── Public routes (exempt from auth middleware) ───────────────────────────────

PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/health",
        "/api/v1/health",
        "/api/v1/auth/login",
        "/api/v1/auth/signup",
        "/api/v1/auth/callback",
        "/api/v1/auth/google/callback",  # Google OAuth redirect — no JWT present
        "/api/v1/auth/refresh",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)

# Path prefixes that are always public (e.g. "/static/").
# Requests under /internal/ bypass Supabase JWT middleware and are protected
# separately by the X-Internal-Key dependency in api/main.py.
PUBLIC_PATH_PREFIXES: tuple[str, ...] = (
    "/internal/",
    "/audio/stream/",  # one-time token auth, no JWT required
)

# ── JWKS cache ────────────────────────────────────────────────────────────────

_JWKS_CACHE: dict[str, dict[str, Any]] = {}
_JWKS_CACHE_TTL_SECONDS: int = 300
_JWKS_LOCK = asyncio.Lock()

_JWKS_FETCH_MAX_ATTEMPTS: int = 2
_JWKS_FETCH_RETRY_DELAY_S: float = 0.5
_JWKS_FETCH_TIMEOUT_S: float = 5.0


def _expected_issuer() -> str:
    settings = get_settings()
    return f"{settings.supabase_url.rstrip('/')}/auth/v1"


def _jwks_url() -> str:
    settings = get_settings()
    return f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"


def _normalize_path(path: str) -> str:
    normalized = path.rstrip("/")
    return normalized or "/"


def _is_public_path(path: str) -> bool:
    if path in PUBLIC_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in PUBLIC_PATH_PREFIXES)


async def _fetch_jwks_once(url: str) -> dict[str, dict[str, Any]]:
    async with httpx.AsyncClient(timeout=_JWKS_FETCH_TIMEOUT_S) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    raw_keys = data.get("keys")
    if not isinstance(raw_keys, list):
        raise AuthError("Invalid JWKS response from Supabase: missing 'keys' list")

    jwks_by_kid: dict[str, dict[str, Any]] = {}
    for key in raw_keys:
        if isinstance(key, dict) and key.get("kid"):
            jwks_by_kid[key["kid"]] = key

    if not jwks_by_kid:
        raise AuthError("Invalid JWKS response from Supabase: no usable keys found")

    return jwks_by_kid


async def _fetch_jwks(force_refresh: bool = False) -> dict[str, dict[str, Any]]:
    url = _jwks_url()
    now = time.time()

    if not force_refresh:
        cached = _JWKS_CACHE.get(url)
        if cached and cached["expires_at"] > now:
            return cast(dict[str, dict[str, Any]], cached["data"])

    async with _JWKS_LOCK:
        if not force_refresh:
            cached = _JWKS_CACHE.get(url)
            if cached and cached["expires_at"] > now:
                return cast(dict[str, dict[str, Any]], cached["data"])

        last_exc: Exception | None = None

        for attempt in range(1, _JWKS_FETCH_MAX_ATTEMPTS + 1):
            try:
                jwks_by_kid = await _fetch_jwks_once(url)

                _JWKS_CACHE[url] = {
                    "data": jwks_by_kid,
                    "expires_at": now + _JWKS_CACHE_TTL_SECONDS,
                }
                return jwks_by_kid

            except AuthError:
                raise

            except Exception as exc:
                last_exc = exc
                if attempt < _JWKS_FETCH_MAX_ATTEMPTS:
                    logger.warning(
                        "JWKS fetch attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt,
                        _JWKS_FETCH_MAX_ATTEMPTS,
                        exc,
                        _JWKS_FETCH_RETRY_DELAY_S,
                    )
                    await asyncio.sleep(_JWKS_FETCH_RETRY_DELAY_S)

        raise AuthError(
            f"Failed to fetch Supabase JWKS after {_JWKS_FETCH_MAX_ATTEMPTS} attempt(s): {last_exc}"
        ) from last_exc


def _get_token_header(token: str) -> dict[str, Any]:
    try:
        return cast(dict[str, Any], pyjwt.get_unverified_header(token))
    except pyjwt.InvalidTokenError as exc:
        raise AuthError(f"Malformed token header: {exc}") from exc


def _decode_with_hs256(token: str) -> dict[str, Any]:
    settings = get_settings()

    if not settings.supabase_jwt_secret:
        raise AuthError("HS256 token received but SUPABASE_JWT_SECRET is not configured")

    try:
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
        raise TokenExpiredError("Token has expired — please re-authenticate") from exc
    except pyjwt.InvalidTokenError as exc:
        raise AuthError(f"Invalid HS256 token: {exc}") from exc


async def _decode_with_jwks(token: str, kid: str) -> dict[str, Any]:
    try:
        jwks_by_kid = await _fetch_jwks()
    except AuthError:
        raise
    except Exception as exc:
        raise AuthError(f"Failed to fetch Supabase JWKS: {exc}") from exc

    matching_key = jwks_by_kid.get(kid)

    if not matching_key:
        logger.info("JWT kid %r not in JWKS cache — refreshing", kid)
        jwks_by_kid = await _fetch_jwks(force_refresh=True)
        matching_key = jwks_by_kid.get(kid)

    if not matching_key:
        raise AuthError(f"No matching signing key found for kid={kid!r}")

    try:
        alg_type = matching_key.get("kty", "RSA")
        if alg_type == "EC":
            from cryptography.hazmat.primitives.asymmetric.ec import EllipticCurvePublicKey
            public_key = algorithms.ECAlgorithm.from_jwk(matching_key)
        else:
            public_key = algorithms.RSAAlgorithm.from_jwk(matching_key)

        return cast(
            dict[str, Any],
            pyjwt.decode(
                token,
                public_key,
                algorithms=["RS256", "ES256"],
                audience="authenticated",
                issuer=_expected_issuer(),
                options={"require": ["exp", "sub", "aud", "iss"]},
            ),
        )
    except pyjwt.ExpiredSignatureError as exc:
        raise TokenExpiredError("Token has expired — please re-authenticate") from exc
    except pyjwt.InvalidTokenError as exc:
        raise AuthError(f"Invalid RS256 token: {exc}") from exc


async def decode_supabase_jwt(token: str) -> dict[str, Any]:
    if not token or not token.strip():
        raise AuthError("Token must not be empty")

    header = _get_token_header(token)
    alg = header.get("alg")
    kid = header.get("kid")

    if alg == "HS256":
        return _decode_with_hs256(token)

    if alg in ("RS256", "ES256"):
        if not kid:
            raise AuthError(f"{alg} token missing 'kid'")
        return await _decode_with_jwks(token, str(kid))

    raise AuthError(f"Unsupported JWT algorithm: {alg!r}")


def _extract_user_id(payload: dict[str, Any]) -> str:
    user_id = str(payload.get("sub", "")).strip()
    if not user_id:
        raise AuthError("Invalid token: missing sub")
    return user_id


def extract_token_from_request(request: Request) -> str | None:
    auth_header = str(request.headers.get("Authorization", ""))
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()
    return None


def extract_token_from_websocket(websocket: WebSocket) -> str | None:
    protocol_header = str(websocket.headers.get("Sec-WebSocket-Protocol", ""))
    for proto in protocol_header.split(","):
        if proto.strip().startswith("bearer."):
            return proto.split("bearer.")[1].strip()

    auth_header = str(websocket.headers.get("Authorization", ""))
    if auth_header.startswith("Bearer "):
        return auth_header[7:].strip()

    return websocket.query_params.get("token")


class SupabaseAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = _normalize_path(request.url.path)

        if request.method == "OPTIONS":
            return await call_next(request)

        if _is_public_path(path):
            return await call_next(request)

        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await call_next(request)

        token = extract_token_from_request(request)
        if not token:
            return JSONResponse(status_code=401, content={"detail": "Missing token"})

        try:
            payload = await decode_supabase_jwt(token)
            request.state.user_id = _extract_user_id(payload)
        except TokenExpiredError:
            return JSONResponse(
                status_code=401,
                content={"detail": "Token has expired — please re-authenticate"},
                headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
            )
        except AuthError:
            return JSONResponse(status_code=401, content={"detail": "Invalid token"})
        except Exception:
            # Infrastructure failure (JWKS fetch down, network error) — not a
            # client auth error. Return 503 so clients don't retry with a new
            # token when the issue is on the server side.
            logger.exception("Auth middleware infrastructure error on %s", request.url.path)
            return JSONResponse(
                status_code=503,
                content={"detail": "Authentication service unavailable"},
            )

        return await call_next(request)


async def authenticate_websocket(websocket: WebSocket) -> tuple[str, dict[str, Any]]:
    token = extract_token_from_websocket(websocket)

    if not token:
        await websocket.close(code=4401, reason="Missing token")
        raise WebSocketAuthError("Missing token")

    payload = await decode_supabase_jwt(token)
    user_id = _extract_user_id(payload)

    return user_id, payload