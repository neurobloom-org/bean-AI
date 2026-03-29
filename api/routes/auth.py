"""BEAN AI — Authentication routes.

Four public routes (no JWT required — listed in PUBLIC_PATHS):
    POST /api/v1/auth/login              — sign in with email + password
    POST /api/v1/auth/signup             — create a new account
    POST /api/v1/auth/refresh            — exchange refresh token for new JWT
    GET  /api/v1/auth/google/callback    — Google OAuth callback (Calendar)

One protected route (JWT required):
    GET  /api/v1/auth/google             — begin Google Calendar OAuth flow

All login/signup/refresh operations are thin wrappers over the Supabase Auth
API. BEAN never handles raw passwords — Supabase Auth owns credential storage.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr

from api.routes._deps import get_current_user_id
from services.supabase_client import get_anon_client
from services.calendar_service import (
    get_authorization_url,
    handle_callback,
    has_calendar_connected,
)
from shared.config import get_settings
from shared.exceptions import CalendarError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ── Request/response models ───────────────────────────────────────────────────


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class SignupRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: str | None = None


class RefreshRequest(BaseModel):
    refresh_token: str


# ── Auth routes ───────────────────────────────────────────────────────────────


@router.post("/login")
async def login(body: LoginRequest) -> dict[str, Any]:
    """Sign in with email and password.

    Delegates to Supabase Auth. Returns the access_token, refresh_token,
    and basic user info. The access_token is a Supabase JWT that must be
    sent as `Authorization: Bearer <token>` on all subsequent requests.
    """
    try:
        client = await get_anon_client()
        response = await client.auth.sign_in_with_password(
            {"email": body.email, "password": body.password}
        )
        if not response.session:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        return {
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
            "token_type": "bearer",
            "expires_in": response.session.expires_in,
            "user": {
                "id": response.user.id if response.user else None,
                "email": response.user.email if response.user else None,
            },
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Login failed for %s: %s", body.email, exc)
        raise HTTPException(status_code=401, detail="Invalid email or password")


@router.post("/signup", status_code=201)
async def signup(body: SignupRequest) -> dict[str, Any]:
    """Create a new user account.

    Delegates to Supabase Auth. Returns the same session shape as /login.
    If email confirmation is enabled in the Supabase project, the session
    will be None and the user must verify their email before logging in.
    """
    try:
        client = await get_anon_client()
        options: dict[str, Any] = {}
        if body.display_name:
            options["data"] = {"display_name": body.display_name}

        response = await client.auth.sign_up(
            {"email": body.email, "password": body.password, "options": options}
        )
        if not response.user:
            raise HTTPException(status_code=400, detail="Signup failed")

        result: dict[str, Any] = {
            "user": {
                "id": response.user.id,
                "email": response.user.email,
            },
        }
        if response.session:
            result["access_token"] = response.session.access_token
            result["refresh_token"] = response.session.refresh_token
            result["token_type"] = "bearer"
            result["expires_in"] = response.session.expires_in
        else:
            result["message"] = "Check your email to confirm your account"

        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Signup failed for %s: %s", body.email, exc)
        raise HTTPException(status_code=400, detail="Signup failed")


@router.post("/refresh")
async def refresh_token(body: RefreshRequest) -> dict[str, Any]:
    """Exchange a refresh token for a new access token.

    Supabase refresh tokens are single-use. A new refresh token is returned
    alongside the new access token — the client must store the latest one.
    """
    try:
        client = await get_anon_client()
        response = await client.auth.refresh_session(body.refresh_token)
        if not response.session:
            raise HTTPException(status_code=401, detail="Invalid or expired refresh token")

        return {
            "access_token": response.session.access_token,
            "refresh_token": response.session.refresh_token,
            "token_type": "bearer",
            "expires_in": response.session.expires_in,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Token refresh failed: %s", exc)
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")


# ── Google Calendar OAuth ─────────────────────────────────────────────────────


@router.get("/google")
async def google_oauth_start(
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict[str, Any]:
    """Begin the Google Calendar OAuth flow for the authenticated user.

    Returns the URL the client should redirect the user to. The frontend
    should open this URL in a browser tab. After the user grants access,
    Google redirects to /api/v1/auth/google/callback which stores the tokens.
    """
    try:
        settings = get_settings()
        if not settings.google_oauth_client_id:
            raise HTTPException(
                status_code=503,
                detail="Google Calendar integration is not configured",
            )
        already_connected = await has_calendar_connected(user_id)
        auth_url = get_authorization_url(user_id)
        return {
            "auth_url": auth_url,
            "already_connected": already_connected,
        }
    except HTTPException:
        raise
    except CalendarError as exc:
        logger.warning("Cannot build Google auth URL for user %s: %s", user_id, exc)
        raise HTTPException(status_code=503, detail="Google Calendar integration unavailable")


@router.get("/google/callback")
async def google_oauth_callback(
    code: str = Query(..., description="Authorization code from Google"),
    state: str = Query(..., description="Signed state token"),
    error: str | None = Query(None, description="Error from Google if user denied access"),
) -> RedirectResponse:
    """Handle the Google OAuth redirect.

    This route is called by Google after the user grants (or denies) access.
    It is exempt from JWT auth middleware (listed in PUBLIC_PATHS) because
    Google does not send a Bearer token.

    On success: stores tokens and redirects to FRONTEND_BASE_URL/calendar-connected
    On failure: redirects to FRONTEND_BASE_URL/calendar-error
    """
    settings = get_settings()
    frontend = settings.frontend_base_url.rstrip("/")

    if error:
        logger.info("Google OAuth denied by user: %s", error)
        return RedirectResponse(url=f"{frontend}/calendar-error?reason=denied")

    try:
        await handle_callback(code=code, state=state)
        return RedirectResponse(url=f"{frontend}/calendar-connected")
    except CalendarError as exc:
        logger.warning("Google OAuth callback failed: %s", exc)
        return RedirectResponse(url=f"{frontend}/calendar-error?reason=failed")
    except Exception as exc:
        logger.error("Unexpected error in Google OAuth callback: %s", exc)
        return RedirectResponse(url=f"{frontend}/calendar-error?reason=error")
