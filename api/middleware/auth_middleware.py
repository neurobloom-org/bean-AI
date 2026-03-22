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
        "/api/v1/auth/refresh",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)

# Path prefixes that are always public (e.g. "/static/").
# Empty by default — add entries here rather than in PUBLIC_PATHS for prefix matching.
PUBLIC_PATH_PREFIXES: tuple[str, ...] = ()

# ── JWKS cache ────────────────────────────────────────────────────────────────

# keyed by JWKS URL → {data: {kid: key_dict}, expires_at: float}
_JWKS_CACHE: dict[str, dict[str, Any]] = {}
_JWKS_CACHE_TTL_SECONDS: int = 300  # 5 minutes
_JWKS_LOCK = asyncio.Lock()

# How many times to retry a transient JWKS fetch failure before giving up.
_JWKS_FETCH_MAX_ATTEMPTS: int = 2
_JWKS_FETCH_RETRY_DELAY_S: float = 0.5

# Timeout for a single JWKS HTTP fetch.
_JWKS_FETCH_TIMEOUT_S: float = 5.0


# ── Internal helpers ──────────────────────────────────────────────────────────


def _expected_issuer() -> str:
    """Return the expected JWT issuer for this Supabase project."""
    settings = get_settings()
    return f"{settings.supabase_url.rstrip('/')}/auth/v1"


def _jwks_url() -> str:
    """Return the JWKS endpoint for this Supabase project."""
    settings = get_settings()
    return f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"


def _normalize_path(path: str) -> str:
    """Normalize request paths so /docs and /docs/ are treated identically."""
    normalized = path.rstrip("/")
    return normalized or "/"


def _is_public_path(path: str) -> bool:
    """Return True if the path requires no authentication."""
    if path in PUBLIC_PATHS:
        return True
    return any(path.startswith(prefix) for prefix in PUBLIC_PATH_PREFIXES)


async def _fetch_jwks_once(url: str) -> dict[str, dict[str, Any]]:
    """Perform one JWKS HTTP fetch and return a kid→key dict.

    Raises:
        AuthError: If the response is malformed or contains no usable keys.
        httpx.HTTPError: On network failure (caller handles retry).
    """
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
    """Fetch and cache the JWKS document from Supabase, with retry.

    Uses a double-checked lock so concurrent requests share one fetch.
    The outer (lock-free) read is safe on CPython's GIL but is documented
    as a fast-path optimisation — the lock is always acquired before writing.

    Args:
        force_refresh: Bypass the cache and fetch unconditionally.

    Returns:
        Dict mapping key ID (kid) → raw JWK dict.

    Raises:
        AuthError: On malformed JWKS response or persistent network failure.
    """
    url = _jwks_url()
    now = time.time()

    # Fast path: return cached data if still fresh (no lock needed for read).
    if not force_refresh:
        cached = _JWKS_CACHE.get(url)
        if cached and cached["expires_at"] > now:
            return cast(dict[str, dict[str, Any]], cached["data"])

    async with _JWKS_LOCK:
        # Re-check under the lock — another coroutine may have populated it.
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
                # Malformed JWKS — retrying won't help.
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
            f"Failed to fetch Supabase JWKS after {_JWKS_FETCH_MAX_ATTEMPTS} "
            f"attempt(s): {last_exc}"
        )


def _get_token_header(token: str) -> dict[str, Any]:
    """Read the unverified JWT header.

    Raises:
        AuthError: If the token header cannot be parsed.
    """
    try:
        return cast(dict[str, Any], pyjwt.get_unverified_header(token))
    except pyjwt.InvalidTokenError as exc:
        raise AuthError(f"Malformed token header: {exc}") from exc


def _decode_with_hs256(token: str) -> dict[str, Any]:
    """Validate a legacy HS256 Supabase JWT using the JWT secret.

    Raises:
        TokenExpiredError: If the token has expired.
        AuthError:         For any other validation failure.
    """
    settings = get_settings()

    if not settings.supabase_jwt_secret:
        raise AuthError(
            "HS256 token received but SUPABASE_JWT_SECRET is not configured"
        )

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
    """Validate a modern RS256 Supabase JWT using the project's JWKS.

    Attempts a cache-refresh if the key ID is not found on first lookup.

    Raises:
        TokenExpiredError: If the token has expired.
        AuthError:         For any other validation failure.
    """
    try:
        jwks_by_kid = await _fetch_jwks()
    except AuthError:
        raise
    except Exception as exc:
        raise AuthError(f"Failed to fetch Supabase JWKS: {exc}") from exc

    matching_key = jwks_by_kid.get(kid)

    if not matching_key:
        # Kid not in cache — might be a newly rotated key. Force refresh once.
        logger.info("JWT kid %r not in JWKS cache — refreshing", kid)
        try:
            jwks_by_kid = await _fetch_jwks(force_refresh=True)
        except AuthError:
            raise
        except Exception as exc:
            raise AuthError(f"Failed to refresh Supabase JWKS: {exc}") from exc

        matching_key = jwks_by_kid.get(kid)

    if not matching_key:
        raise AuthError(
            f"No matching signing key found for kid={kid!r}. "
            "The token may have been signed with a rotated key."
        )

    try:
        public_key = cast(RSAPublicKey, algorithms.RSAAlgorithm.from_jwk(matching_key))
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
        raise TokenExpiredError("Token has expired — please re-authenticate") from exc
    except pyjwt.InvalidTokenError as exc:
        raise AuthError(f"Invalid RS256 token: {exc}") from exc


# ── Public decode API ─────────────────────────────────────────────────────────


async def decode_supabase_jwt(token: str) -> dict[str, Any]:
    """Decode and validate a Supabase-issued JWT.

    Dispatches to HS256 or RS256 validation based on the token header.

    Args:
        token: Raw JWT string (without "Bearer " prefix).

    Returns:
        Decoded payload dict. Always contains "sub" (user ID).

    Raises:
        TokenExpiredError: The token signature is valid but it has expired.
        AuthError:         Any other validation failure (bad signature,
                           unsupported algorithm, malformed header, etc.).
    """
    if not token or not token.strip():
        raise AuthError("Token must not be empty")

    header = _get_token_header(token)
    alg = header.get("alg")
    kid = header.get("kid")

    if alg == "HS256":
        return _decode_with_hs256(token)

    if alg == "RS256":
        if not kid:
            raise AuthError("RS256 token is missing the 'kid' (key ID) header field")
        return await _decode_with_jwks(token, str(kid))

    raise AuthError(
        f"Unsupported JWT algorithm: {alg!r}. "
        "Expected 'HS256' (legacy) or 'RS256' (modern Supabase)."
    )


def _extract_user_id(payload: dict[str, Any]) -> str:
    """Extract and validate the user ID (sub claim) from a decoded payload.

    Raises:
        AuthError: If 'sub' is missing or empty.
    """
    try:
        sub = payload["sub"]
    except KeyError as exc:
        raise AuthError(
            "Token payload is missing required 'sub' (subject) claim"
        ) from exc

    user_id = str(sub).strip()
    if not user_id:
        raise AuthError("Token 'sub' claim is empty — cannot identify user")

    return user_id


# ── Token extraction helpers ──────────────────────────────────────────────────


def extract_token_from_request(request: Request) -> str | None:
    """Extract a JWT from the HTTP Authorization header.

    Returns the raw token string (without "Bearer " prefix), or None if
    the header is absent or malformed.
    """
    auth_header = str(request.headers.get("Authorization", ""))
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        return token or None
    return None


def extract_token_from_websocket(websocket: WebSocket) -> str | None:
    """Extract a JWT from a WebSocket connection.

    Checks in order:
      1. Sec-WebSocket-Protocol header — ESP32 sends "bearer.<token>"
      2. Authorization header — "Bearer <token>" (browser / test clients)
      3. Query parameter — "?token=<token>" (fallback for clients that
         cannot set custom headers, e.g. some browser WS APIs)

    Returns the raw token string, or None if not found.
    """
    # 1. Sec-WebSocket-Protocol: bearer.<token>
    protocol_header = str(websocket.headers.get("Sec-WebSocket-Protocol", ""))
    for proto in protocol_header.split(","):
        proto = proto.strip()
        if proto.startswith("bearer."):
            token = proto[len("bearer.") :].strip()
            if token:
                return token

    # 2. Authorization: Bearer <token>
    auth_header = str(websocket.headers.get("Authorization", ""))
    if auth_header.startswith("Bearer "):
        token = auth_header[7:].strip()
        if token:
            return token

    # 3. Query parameter: ?token=<token>
    token = websocket.query_params.get("token", "").strip()
    if token:
        return token

    return None


# ── HTTP middleware ───────────────────────────────────────────────────────────


class SupabaseAuthMiddleware(BaseHTTPMiddleware):  # type: ignore[misc]
    """Validate a Supabase JWT on every non-public HTTP request.

    Skips:
      - OPTIONS preflight requests (CORS)
      - Public paths (PUBLIC_PATHS / PUBLIC_PATH_PREFIXES)
      - WebSocket upgrade requests (auth handled by authenticate_websocket)

    On success, attaches to request.state:
      - user_id     (str)
      - email       (str | None)
      - role        (str | None)
      - jwt_payload (dict)
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = _normalize_path(request.url.path)

        # Always pass OPTIONS through — CORS preflight must not be gated.
        if request.method.upper() == "OPTIONS":
            return await call_next(request)

        # Public routes need no token.
        if _is_public_path(path):
            return await call_next(request)

        # WebSocket upgrades bypass HTTP middleware — auth happens in
        # authenticate_websocket() which is called by the WS route handler.
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return await call_next(request)

        token = extract_token_from_request(request)
        if not token:
            return JSONResponse(
                status_code=401,
                content={
                    "detail": (
                        "Authorization header is required. "
                        "Use: Authorization: Bearer <token>"
                    )
                },
            )

        try:
            payload = await decode_supabase_jwt(token)
            user_id = _extract_user_id(payload)

            request.state.user_id = user_id
            request.state.email = payload.get("email")
            request.state.role = payload.get("role")
            request.state.jwt_payload = payload

        except TokenExpiredError as exc:
            logger.info("Auth rejected — expired token [path=%s]: %s", path, exc)
            return JSONResponse(
                status_code=401,
                content={"detail": str(exc)},
                headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
            )

        except AuthError as exc:
            logger.warning("Auth rejected [path=%s]: %s", path, exc)
            return JSONResponse(
                status_code=401,
                content={"detail": str(exc)},
                headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
            )

        return await call_next(request)


# ── WebSocket auth helper ─────────────────────────────────────────────────────


async def authenticate_websocket(websocket: WebSocket) -> tuple[str, dict[str, Any]]:
    """Authenticate an incoming WebSocket connection.

    Extracts the JWT, validates it, and returns (user_id, jwt_payload).

    On failure, closes the WebSocket with a 4401 close code before raising
    so the ESP32 knows to reconnect with a fresh token.

    Args:
        websocket: The incoming WebSocket connection (not yet accepted).

    Returns:
        (user_id, payload) — user_id is always a non-empty string.

    Raises:
        WebSocketAuthError: If the token is missing, expired, or invalid.
                            The WebSocket is always closed before this raises.
    """
    token = extract_token_from_websocket(websocket)

    if not token:
        await websocket.close(
            code=4401,
            reason=(
                "Authentication token is required. "
                "Send via Sec-WebSocket-Protocol: bearer.<token>"
            ),
        )
        raise WebSocketAuthError("No authentication token provided")

    try:
        payload = await decode_supabase_jwt(token)
        user_id = _extract_user_id(payload)

    except TokenExpiredError as exc:
        logger.info("WebSocket auth rejected — expired token: %s", exc)
        await websocket.close(
            code=4401,
            reason="Token has expired — please re-authenticate and reconnect",
        )
        raise WebSocketAuthError(str(exc)) from exc

    except AuthError as exc:
        logger.warning("WebSocket auth rejected: %s", exc)
        await websocket.close(code=4401, reason=str(exc))
        raise WebSocketAuthError(str(exc)) from exc

    logger.debug("WebSocket authenticated: user=%s", user_id[:8])
    return user_id, payload
