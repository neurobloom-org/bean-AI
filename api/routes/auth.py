"""BEAN AI v1 — Auth routes.

Handles:
- Supabase email/password auth
- token refresh / logout
- Google OAuth for Calendar integration

Security notes:
- Google OAuth state is signed with a dedicated app secret and bound to the
  initiating browser via cookie
- Callback redirects back to the frontend instead of returning raw JSON
- Profile creation should be handled by a DB trigger on auth.users
"""

from __future__ import annotations

import logging
import urllib.parse
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import httpx
import jwt as pyjwt
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, EmailStr, Field

from api.middleware.auth_middleware import decode_supabase_jwt
from services.supabase_client import (
    get_anon_client,
    get_service_client,
    get_authed_client,
)
from shared.config import get_settings

try:
    # Supabase Python auth errors are wrapped in AuthError according to docs.
    from gotrue.errors import AuthError
except Exception:  # pragma: no cover
    AuthError = Exception  


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

bearer_scheme = HTTPBearer(auto_error=False)

GOOGLE_PROVIDER = "google_calendar"
GOOGLE_CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar"
GOOGLE_OAUTH_STATE_COOKIE = "google_oauth_state"
GOOGLE_OAUTH_STATE_TTL_SECONDS = 600

# Reused HTTP client for connection pooling.
# Close this on app shutdown if you manage lifespan events centrally.
OAUTH_HTTP_CLIENT = httpx.AsyncClient(timeout=10.0)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)


class SignupRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: str | None = Field(default=None, max_length=100)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(min_length=1)


def _frontend_oauth_redirect_url(success: bool, message: str | None = None) -> str:
    """Build the frontend redirect URL after Google OAuth completes."""
    settings = get_settings()
    base_url = (
        getattr(settings, "frontend_base_url", "").rstrip("/")
        or "http://localhost:3000"
    )

    params = {"calendar": "success" if success else "error"}
    if message:
        params["message"] = message

    return f"{base_url}/settings/integrations?{urllib.parse.urlencode(params)}"


def _oauth_cookie_domain() -> str | None:
    """Return the configured cookie domain for OAuth state, if any."""
    settings = get_settings()
    return getattr(settings, "cookie_domain", None)


def _oauth_state_secret() -> str:
    """Return the dedicated secret for OAuth state signing."""
    settings = get_settings()
    secret = getattr(settings, "oauth_state_secret", None)
    if not secret:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OAUTH_STATE_SECRET is not configured",
        )
    return str(secret)


def _build_google_state_token(user_id: str) -> str:
    """Create a signed short-lived JWT for Google OAuth state."""
    secret = _oauth_state_secret()
    now = datetime.now(UTC)

    payload = {
        "sub": user_id,
        "purpose": "google_oauth_state",
        "iat": int(now.timestamp()),
        "exp": int(
            (now + timedelta(seconds=GOOGLE_OAUTH_STATE_TTL_SECONDS)).timestamp()
        ),
    }

    return pyjwt.encode(payload, secret, algorithm="HS256")


def _decode_google_state_token(state_token: str) -> str:
    """Decode and validate the signed Google OAuth state token."""
    secret = _oauth_state_secret()

    try:
        payload = pyjwt.decode(
            state_token,
            secret,
            algorithms=["HS256"],
            options={"require": ["sub", "purpose", "iat", "exp"]},
        )
    except pyjwt.ExpiredSignatureError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="OAuth state expired",
        ) from exc
    except pyjwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid OAuth state",
        ) from exc

    if payload.get("purpose") != "google_oauth_state":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid OAuth state purpose",
        )

    user_id = str(payload.get("sub") or "")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid OAuth state payload",
        )

    return user_id


async def get_current_bearer_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> str:
    """Return the raw bearer token."""
    if not credentials or not credentials.credentials.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is missing or invalid",
        )
    return credentials.credentials.strip()


async def get_current_user_id(
    token: Annotated[str, Depends(get_current_bearer_token)],
) -> str:
    """Validate the bearer token and return user_id."""
    try:
        payload = await decode_supabase_jwt(token)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc

    user_id = str(payload.get("sub") or "")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
        )

    return user_id


@router.post("/login")
async def login(body: LoginRequest) -> dict[str, Any]:
    """Authenticate a user with Supabase email/password auth."""
    try:
        client = await get_anon_client()
        result = await client.auth.sign_in_with_password(
            {"email": body.email, "password": body.password}
        )

        if not result.session or not result.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        session = result.session
        return {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
            "expires_in": session.expires_in,
            "user_id": result.user.id,
        }
    except HTTPException:
        raise
    except AuthError as exc:
        logger.warning("Login failed for %s: %s", body.email, exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.warning("Login failed for %s: %s", body.email, exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        ) from exc


@router.post("/signup")
async def signup(body: SignupRequest) -> dict[str, Any]:
    """Create a new Supabase user.

    Profile row creation should be handled by a DB trigger on auth.users.
    display_name is passed through user metadata for the trigger to consume.
    """
    try:
        client = await get_anon_client()
        result = await client.auth.sign_up(
            {
                "email": body.email,
                "password": body.password,
                "options": {
                    "data": {
                        "display_name": body.display_name,
                    }
                },
            }
        )

        if not result.user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Signup failed",
            )

        return {
            "user_id": result.user.id,
            "email": result.user.email,
            "message": "Account created. Check your email to confirm.",
        }
    except HTTPException:
        raise
    except AuthError as exc:
        logger.error("Signup failed for %s: %s", body.email, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error("Signup failed for %s: %s", body.email, exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to create account",
        ) from exc


@router.post("/refresh")
async def refresh_token(body: RefreshRequest) -> dict[str, Any]:
    """Refresh a Supabase session using a refresh token."""
    try:
        client = await get_anon_client()
        result = await client.auth.refresh_session(body.refresh_token)

        if not result.session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token",
            )

        session = result.session
        return {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
            "expires_in": session.expires_in,
        }
    except HTTPException:
        raise
    except AuthError as exc:
        logger.warning("Refresh token failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.warning("Refresh token failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        ) from exc


@router.post("/logout")
async def logout(
    token: Annotated[str, Depends(get_current_bearer_token)],
) -> dict[str, str]:
    """Log out the current Supabase user session."""
    try:
        client = await get_authed_client(token)
        await client.auth.sign_out()
    except Exception as exc:
        logger.warning("Logout failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to log out",
        ) from exc

    return {"message": "Logged out"}


@router.get("/google")
async def google_oauth_start(
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> RedirectResponse:
    """Redirect the authenticated user to Google's OAuth consent screen."""
    settings = get_settings()
    state_token = _build_google_state_token(user_id)

    params = {
        "client_id": settings.google_oauth_client_id,
        "redirect_uri": settings.google_oauth_redirect_uri,
        "response_type": "code",
        "scope": GOOGLE_CALENDAR_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "state": state_token,
    }

    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(
        params
    )

    response = RedirectResponse(url=url)
    response.set_cookie(
        key=GOOGLE_OAUTH_STATE_COOKIE,
        value=state_token,
        max_age=GOOGLE_OAUTH_STATE_TTL_SECONDS,
        httponly=True,
        secure=True,
        samesite="lax",
        domain=_oauth_cookie_domain(),
    )
    return response


@router.get("/google/callback")
async def google_oauth_callback(
    request: Request,
    code: str | None = Query(default=None),
    state: str | None = Query(default=None),
    error: str | None = Query(default=None),
) -> RedirectResponse:
    """Handle Google OAuth callback and store Calendar tokens securely."""
    settings = get_settings()
    cookie_domain = _oauth_cookie_domain()

    if error:
        response = RedirectResponse(
            url=_frontend_oauth_redirect_url(False, error),
            status_code=status.HTTP_303_SEE_OTHER,
        )
        response.delete_cookie(GOOGLE_OAUTH_STATE_COOKIE, domain=cookie_domain)
        return response

    if not code or not state:
        response = RedirectResponse(
            url=_frontend_oauth_redirect_url(False, "missing_code_or_state"),
            status_code=status.HTTP_303_SEE_OTHER,
        )
        response.delete_cookie(GOOGLE_OAUTH_STATE_COOKIE, domain=cookie_domain)
        return response

    cookie_state = request.cookies.get(GOOGLE_OAUTH_STATE_COOKIE)
    if not cookie_state or cookie_state != state:
        response = RedirectResponse(
            url=_frontend_oauth_redirect_url(False, "invalid_state"),
            status_code=status.HTTP_303_SEE_OTHER,
        )
        response.delete_cookie(GOOGLE_OAUTH_STATE_COOKIE, domain=cookie_domain)
        return response

    try:
        user_id = _decode_google_state_token(state)
    except HTTPException as exc:
        response = RedirectResponse(
            url=_frontend_oauth_redirect_url(False, exc.detail),
            status_code=status.HTTP_303_SEE_OTHER,
        )
        response.delete_cookie(GOOGLE_OAUTH_STATE_COOKIE, domain=cookie_domain)
        return response

    try:
        token_response = await OAUTH_HTTP_CLIENT.post(
            "https://oauth2.googleapis.com/token",
            data={
                "client_id": settings.google_oauth_client_id,
                "client_secret": settings.google_oauth_client_secret,
                "code": code,
                "redirect_uri": settings.google_oauth_redirect_uri,
                "grant_type": "authorization_code",
            },
        )
        token_response.raise_for_status()
        token_data = token_response.json()
    except Exception as exc:
        logger.error("Google token exchange failed for user=%s: %s", user_id, exc)
        response = RedirectResponse(
            url=_frontend_oauth_redirect_url(False, "token_exchange_failed"),
            status_code=status.HTTP_303_SEE_OTHER,
        )
        response.delete_cookie(GOOGLE_OAUTH_STATE_COOKIE, domain=cookie_domain)
        return response

    access_token = token_data.get("access_token")
    granted_scope = token_data.get("scope", "")

    if not access_token:
        response = RedirectResponse(
            url=_frontend_oauth_redirect_url(False, "missing_access_token"),
            status_code=status.HTTP_303_SEE_OTHER,
        )
        response.delete_cookie(GOOGLE_OAUTH_STATE_COOKIE, domain=cookie_domain)
        return response

    granted_scopes = set(granted_scope.split())
    if GOOGLE_CALENDAR_SCOPE not in granted_scopes:
        logger.warning(
            "Google OAuth completed without required calendar scope for user=%s scopes=%s",
            user_id,
            granted_scope,
        )
        response = RedirectResponse(
            url=_frontend_oauth_redirect_url(False, "calendar_scope_not_granted"),
            status_code=status.HTTP_303_SEE_OTHER,
        )
        response.delete_cookie(GOOGLE_OAUTH_STATE_COOKIE, domain=cookie_domain)
        return response

    expires_at = (
        datetime.now(UTC) + timedelta(seconds=token_data.get("expires_in", 3600))
    ).isoformat()

    try:
        client = await get_service_client()

        existing = (
            await client.table("oauth_tokens")
            .select("refresh_token")
            .eq("user_id", user_id)
            .eq("provider", GOOGLE_PROVIDER)
            .maybe_single()
            .execute()
        )

        refresh_token = token_data.get("refresh_token")
        if not refresh_token and existing.data:
            refresh_token = existing.data.get("refresh_token")

        await (
            client.table("oauth_tokens")
            .upsert(
                {
                    "user_id": user_id,
                    "provider": GOOGLE_PROVIDER,
                    "access_token": access_token,
                    "refresh_token": refresh_token,
                    "expires_at": expires_at,
                    "scope": granted_scope,
                },
                on_conflict="user_id,provider",
            )
            .execute()
        )

        logger.info("Google Calendar connected for user %s", user_id[:8])
    except Exception as exc:
        logger.error("Failed to store Google tokens for user=%s: %s", user_id, exc)
        response = RedirectResponse(
            url=_frontend_oauth_redirect_url(False, "save_failed"),
            status_code=status.HTTP_303_SEE_OTHER,
        )
        response.delete_cookie(GOOGLE_OAUTH_STATE_COOKIE, domain=cookie_domain)
        return response

    response = RedirectResponse(
        url=_frontend_oauth_redirect_url(True),
        status_code=status.HTTP_303_SEE_OTHER,
    )
    response.delete_cookie(GOOGLE_OAUTH_STATE_COOKIE, domain=cookie_domain)
    return response


@router.delete("/google")
async def google_oauth_disconnect(
    user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
) -> dict[str, str]:
    """Disconnect the authenticated user's Google Calendar integration."""
    try:
        client = await get_authed_client(token)
        await (
            client.table("oauth_tokens")
            .delete()
            .eq("user_id", user_id)
            .eq("provider", GOOGLE_PROVIDER)
            .execute()
        )
    except Exception as exc:
        logger.error("Failed to remove Google tokens for user=%s: %s", user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to disconnect calendar",
        ) from exc

    return {"message": "Google Calendar disconnected"}


@router.get("/google/status")
async def google_oauth_status(
    user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
) -> dict[str, Any]:
    """Return whether Google Calendar is connected for the authenticated user."""
    try:
        client = await get_authed_client(token)
        result = (
            await client.table("oauth_tokens")
            .select("expires_at, scope")
            .eq("user_id", user_id)
            .eq("provider", GOOGLE_PROVIDER)
            .maybe_single()
            .execute()
        )

        connected = result.data is not None
        return {
            "connected": connected,
            "expires_at": result.data.get("expires_at") if connected else None,
            "scope": result.data.get("scope") if connected else None,
        }
    except Exception as exc:
        logger.error(
            "Failed to fetch Google OAuth status for user=%s: %s",
            user_id,
            exc,
        )
        return {"connected": False, "expires_at": None, "scope": None}
