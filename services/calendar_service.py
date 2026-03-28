"""BEAN AI — Google Calendar OAuth service.

Handles the full Google Calendar OAuth 2.0 flow using httpx directly
(no google-auth-oauthlib dependency — keeps the dep tree light).

OAuth flow:
    1. get_authorization_url(user_id) → signed URL to redirect the user to Google
    2. handle_callback(code, state)   → exchange code for tokens, store in DB
    3. create_calendar_event(...)     → create event using stored access token

Calendar sync is best-effort — create_calendar_event() returns None on failure
rather than raising, so task creation is never blocked by Calendar being down.

Token storage:
    Tokens are upserted into the oauth_tokens table (provider='google_calendar').
    Supabase encrypts the table at rest. Refresh tokens are stored to allow
    silent renewal without re-prompting the user.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import httpx

from services.supabase_client import get_service_client
from shared.config import get_settings
from shared.exceptions import CalendarError

logger = logging.getLogger(__name__)

_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_CALENDAR_API_URL = "https://www.googleapis.com/calendar/v3"
_CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar.events"

# State tokens expire after 10 minutes (Google recommends short-lived state)
_STATE_TTL_SECONDS: int = 600


# ── State token helpers (HMAC-signed, no server-side session needed) ──────────


def _sign_state(user_id: str) -> str:
    """Encode user_id + timestamp into a signed state token for the OAuth flow."""
    settings = get_settings()
    payload = json.dumps({"user_id": user_id, "ts": int(datetime.now(UTC).timestamp())})
    sig = hmac.new(
        settings.oauth_state_secret.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()
    import base64
    return base64.urlsafe_b64encode(f"{payload}|{sig}".encode()).decode().rstrip("=")


def _verify_state(state: str) -> str:
    """Verify a signed state token and return the user_id.

    Raises CalendarError if the signature is invalid or the token has expired.
    """
    settings = get_settings()
    try:
        import base64
        # Restore stripped padding
        padded = state + "=" * (4 - len(state) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode()).decode()
        payload_str, received_sig = decoded.rsplit("|", 1)

        expected_sig = hmac.new(
            settings.oauth_state_secret.encode(),
            payload_str.encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(received_sig, expected_sig):
            raise CalendarError("Invalid OAuth state signature")

        payload = json.loads(payload_str)
        age = int(datetime.now(UTC).timestamp()) - int(payload.get("ts", 0))
        if age > _STATE_TTL_SECONDS:
            raise CalendarError(f"OAuth state expired ({age}s old, max {_STATE_TTL_SECONDS}s)")

        return str(payload["user_id"])

    except CalendarError:
        raise
    except Exception as exc:
        raise CalendarError(f"Could not decode OAuth state: {exc}") from exc


# ── Public API ────────────────────────────────────────────────────────────────


def get_authorization_url(user_id: str) -> str:
    """Build the Google OAuth URL to redirect the authenticated user to.

    The user_id is embedded in a signed state token — no server-side session
    required. The callback handler recovers the user_id by verifying the signature.

    Raises CalendarError if GOOGLE_OAUTH_CLIENT_ID is not configured.
    """
    settings = get_settings()
    if not settings.google_oauth_client_id:
        raise CalendarError("GOOGLE_OAUTH_CLIENT_ID is not configured")

    params = {
        "client_id": settings.google_oauth_client_id,
        "redirect_uri": settings.google_oauth_redirect_uri,
        "response_type": "code",
        "scope": _CALENDAR_SCOPE,
        "access_type": "offline",
        "prompt": "consent",   # always request refresh_token
        "state": _sign_state(user_id),
    }
    return f"{_GOOGLE_AUTH_URL}?{urlencode(params)}"


async def handle_callback(code: str, state: str) -> str:
    """Exchange the authorization code for tokens and store them in the DB.

    Called by the /api/v1/auth/google/callback route after Google redirects back.
    Returns the user_id on success. Raises CalendarError on any failure.
    """
    user_id = _verify_state(state)
    settings = get_settings()

    if not settings.google_oauth_client_id or not settings.google_oauth_client_secret:
        raise CalendarError("Google OAuth client credentials are not configured")

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "code": code,
                "client_id": settings.google_oauth_client_id,
                "client_secret": settings.google_oauth_client_secret,
                "redirect_uri": settings.google_oauth_redirect_uri,
                "grant_type": "authorization_code",
            },
        )

    if resp.status_code != 200:
        raise CalendarError(f"Google token exchange failed: HTTP {resp.status_code}")

    token_data: dict[str, Any] = resp.json()
    if "error" in token_data:
        raise CalendarError(f"Google token exchange error: {token_data['error']}")

    await _store_tokens(user_id, token_data)
    logger.info("Google Calendar tokens stored for user %s", user_id)
    return user_id


async def has_calendar_connected(user_id: str) -> bool:
    """Return True if the user has a stored Google Calendar token."""
    try:
        client = await get_service_client()
        result = (
            await client.table("oauth_tokens")
            .select("id")
            .eq("user_id", user_id)
            .eq("provider", "google_calendar")
            .maybe_single()
            .execute()
        )
        return bool(result and result.data)
    except Exception:
        return False


async def create_calendar_event(
    user_id: str,
    title: str,
    due_at: datetime,
    description: str | None = None,
) -> str | None:
    """Create a Google Calendar event for a task reminder.

    Returns the Google event ID on success, None on any failure.
    Non-fatal by design — callers should proceed even when this returns None.
    """
    try:
        access_token = await _get_valid_access_token(user_id)

        if due_at.tzinfo is None:
            due_at = due_at.replace(tzinfo=UTC)

        event_body: dict[str, Any] = {
            "summary": title,
            "start": {"dateTime": due_at.isoformat()},
            "end": {"dateTime": (due_at + timedelta(hours=1)).isoformat()},
        }
        if description:
            event_body["description"] = description

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{_CALENDAR_API_URL}/calendars/primary/events",
                json=event_body,
                headers={"Authorization": f"Bearer {access_token}"},
            )

        if resp.status_code not in (200, 201):
            logger.warning(
                "Calendar event creation returned HTTP %d for user %s",
                resp.status_code,
                user_id,
            )
            return None

        event_id = str(resp.json().get("id", ""))
        logger.info("Calendar event created [user=%s event=%s]", user_id, event_id)
        return event_id or None

    except CalendarError as exc:
        logger.warning("Calendar not connected for user %s: %s", user_id, exc)
        return None
    except Exception as exc:
        logger.warning("Calendar event creation failed [user=%s]: %s", user_id, exc)
        return None


# ── Internal token management ─────────────────────────────────────────────────


async def _store_tokens(user_id: str, token_data: dict[str, Any]) -> None:
    """Upsert OAuth tokens into the oauth_tokens table."""
    expires_in = int(token_data.get("expires_in", 3600))
    expires_at = (datetime.now(UTC) + timedelta(seconds=expires_in)).isoformat()

    payload: dict[str, Any] = {
        "user_id": user_id,
        "provider": "google_calendar",
        "access_token": token_data["access_token"],
        "expires_at": expires_at,
        "scope": token_data.get("scope"),
        "updated_at": datetime.now(UTC).isoformat(),
    }
    # Only overwrite refresh_token if Google sent a new one
    if token_data.get("refresh_token"):
        payload["refresh_token"] = token_data["refresh_token"]

    client = await get_service_client()
    await (
        client.table("oauth_tokens")
        .upsert(payload, on_conflict="user_id,provider")
        .execute()
    )


async def _get_valid_access_token(user_id: str) -> str:
    """Return a valid access token, transparently refreshing if it has expired.

    Raises CalendarError if no tokens are stored or refresh fails.
    """
    client = await get_service_client()
    result = (
        await client.table("oauth_tokens")
        .select("access_token, refresh_token, expires_at")
        .eq("user_id", user_id)
        .eq("provider", "google_calendar")
        .maybe_single()
        .execute()
    )

    if not result or not result.data:
        raise CalendarError(f"No Google Calendar tokens found for user {user_id}")

    row = result.data
    access_token: str = row["access_token"]
    refresh_token: str | None = row.get("refresh_token")
    expires_at_str: str | None = row.get("expires_at")

    # Refresh if expired (60s buffer to avoid races)
    if expires_at_str:
        expires_at = datetime.fromisoformat(expires_at_str)
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)
        if datetime.now(UTC) >= expires_at - timedelta(seconds=60):
            if not refresh_token:
                raise CalendarError("Access token expired and no refresh token stored")
            access_token = await _refresh_access_token(user_id, refresh_token)

    return access_token


async def _refresh_access_token(user_id: str, refresh_token: str) -> str:
    """Exchange a refresh token for a new access token."""
    settings = get_settings()

    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.post(
            _GOOGLE_TOKEN_URL,
            data={
                "refresh_token": refresh_token,
                "client_id": settings.google_oauth_client_id,
                "client_secret": settings.google_oauth_client_secret,
                "grant_type": "refresh_token",
            },
        )

    if resp.status_code != 200:
        raise CalendarError(f"Token refresh failed: HTTP {resp.status_code}")

    token_data = resp.json()
    if "error" in token_data:
        raise CalendarError(f"Token refresh error: {token_data['error']}")

    # Preserve the existing refresh_token (Google doesn't re-send it on refresh)
    await _store_tokens(user_id, {**token_data, "refresh_token": refresh_token})
    return str(token_data["access_token"])
