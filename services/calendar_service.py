"""BEAN AI v5 — Google Calendar service.

Wraps the Google Calendar API for task/reminder management.
OAuth tokens are stored in the oauth_tokens table in Supabase.

Flow:
  1. User connects their Google account via /api/v1/auth/google
  2. Tokens are stored in oauth_tokens table (encrypted at rest by Supabase)
  3. TaskAgent retrieves token via get_calendar_token(user_id)
  4. CalendarService uses the token to create/list/delete events
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from shared.config import get_settings
from shared.exceptions import CalendarError

logger = logging.getLogger(__name__)


async def get_calendar_token(user_id: str) -> str | None:
    """Retrieve the user's Google Calendar OAuth access token from Supabase.

    Returns None if the user hasn't connected their Google account.
    """
    try:
        from services.supabase_client import get_service_client

        client = await get_service_client()
        result = (
            await client.table("oauth_tokens")
            .select("access_token, expires_at, refresh_token")
            .eq("user_id", user_id)
            .eq("provider", "google_calendar")
            .maybe_single()
            .execute()
        )

        if not result.data:
            return None

        token_data = result.data
        access_token = token_data.get("access_token")
        expires_at = token_data.get("expires_at")
        refresh_token = token_data.get("refresh_token")

        # Check if token is expired
        if expires_at:
            expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            if datetime.now(UTC) >= expiry - timedelta(minutes=5):
                # Token expired — try to refresh
                if refresh_token:
                    access_token = await _refresh_google_token(user_id, refresh_token)
                else:
                    return None

        return access_token

    except Exception as exc:
        logger.error("Failed to retrieve calendar token for user %s: %s", user_id, exc)
        return None


async def _refresh_google_token(user_id: str, refresh_token: str) -> str | None:
    """Refresh an expired Google OAuth token."""
    settings = get_settings()
    try:
        import httpx

        async with httpx.AsyncClient() as http:
            response = await http.post(
                "https://oauth2.googleapis.com/token",
                data={
                    "client_id": settings.google_oauth_client_id,
                    "client_secret": settings.google_oauth_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            response.raise_for_status()
            token_data = response.json()

        new_access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        expires_at = (datetime.now(UTC) + timedelta(seconds=expires_in)).isoformat()

        from services.supabase_client import get_service_client

        client = await get_service_client()
        await (
            client.table("oauth_tokens")
            .update({"access_token": new_access_token, "expires_at": expires_at})
            .eq("user_id", user_id)
            .eq("provider", "google_calendar")
            .execute()
        )

        logger.info("Google token refreshed for user %s", user_id)
        return new_access_token

    except Exception as exc:
        logger.error("Token refresh failed for user %s: %s", user_id, exc)
        return None


class CalendarService:
    """Wrapper around the Google Calendar REST API."""

    CALENDAR_API = "https://www.googleapis.com/calendar/v3"

    def __init__(self, access_token: str):
        self._token = access_token
        self._headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

    async def find_or_create_bean_calendar(self) -> str:
        """Find the 'BEAN Reminders' calendar or create it."""
        import httpx

        async with httpx.AsyncClient() as http:
            # List user's calendars
            response = await http.get(
                f"{self.CALENDAR_API}/users/me/calendarList",
                headers=self._headers,
            )
            response.raise_for_status()
            calendars = response.json().get("items", [])

            for cal in calendars:
                if cal.get("summary") == "BEAN Reminders":
                    return cal["id"]

            # Create the calendar
            create_response = await http.post(
                f"{self.CALENDAR_API}/calendars",
                headers=self._headers,
                json={"summary": "BEAN Reminders", "timeZone": "UTC"},
            )
            create_response.raise_for_status()
            return create_response.json()["id"]

    async def create_event(
        self,
        calendar_id: str,
        title: str,
        event_time: datetime,
        description: str | None = None,
        reminder_minutes: int = 10,
    ) -> dict:
        """Create a calendar event with a popup reminder."""
        import httpx

        event_body = {
            "summary": title,
            "description": description or "",
            "start": {
                "dateTime": event_time.isoformat(),
                "timeZone": "UTC",
            },
            "end": {
                "dateTime": (event_time + timedelta(minutes=30)).isoformat(),
                "timeZone": "UTC",
            },
            "reminders": {
                "useDefault": False,
                "overrides": [
                    {"method": "popup", "minutes": reminder_minutes},
                ],
            },
        }

        async with httpx.AsyncClient() as http:
            response = await http.post(
                f"{self.CALENDAR_API}/calendars/{calendar_id}/events",
                headers=self._headers,
                json=event_body,
            )
            response.raise_for_status()
            return response.json()

    async def list_events(
        self,
        calendar_id: str,
        max_results: int = 5,
    ) -> list[dict]:
        """List upcoming events in the calendar."""
        import httpx

        now = datetime.now(UTC).isoformat()
        async with httpx.AsyncClient() as http:
            response = await http.get(
                f"{self.CALENDAR_API}/calendars/{calendar_id}/events",
                headers=self._headers,
                params={
                    "timeMin": now,
                    "maxResults": max_results,
                    "singleEvents": True,
                    "orderBy": "startTime",
                },
            )
            response.raise_for_status()
            return response.json().get("items", [])

    async def delete_event(self, calendar_id: str, event_id: str) -> None:
        """Delete a calendar event."""
        import httpx

        async with httpx.AsyncClient() as http:
            response = await http.delete(
                f"{self.CALENDAR_API}/calendars/{calendar_id}/events/{event_id}",
                headers=self._headers,
            )
            if response.status_code not in (200, 204):
                raise CalendarError(f"Delete failed: {response.status_code}")
