"""BEAN AI — Supabase async client factory.

Replaces:
  - services/database.py      (SQLAlchemy + asyncpg)
  - services/redis_client.py  (Redis)
  - services/cache_service.py (Redis-based cache)

Two clients:
  - anon_client    → respects Row Level Security (use for user-scoped requests)
  - service_client → bypasses RLS (ONLY for background jobs & admin operations)
"""

import logging
from typing import Any

from supabase import AsyncClient, acreate_client

from shared.config import get_settings

logger = logging.getLogger(__name__)

_anon_client: AsyncClient | None = None
_service_client: AsyncClient | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Client factories
# ─────────────────────────────────────────────────────────────────────────────


async def get_anon_client() -> AsyncClient:
    """Return the anon-key client (respects RLS).

    Use for all user-initiated requests — Supabase enforces that users
    can only access their own rows via RLS policies.
    """
    global _anon_client
    if _anon_client is None:
        settings = get_settings()
        _anon_client = await acreate_client(
            settings.supabase_url,
            settings.supabase_anon_key,
        )
        logger.info("Supabase anon client initialised (RLS enforced)")
    return _anon_client


async def get_service_client() -> AsyncClient:
    """Return the service-role client (bypasses RLS).

    ⚠️  ONLY use for:
      - Background jobs (transcript purge, session cleanup)
      - Admin operations (GDPR deletion, alert broadcasting)
      - Health checks

    NEVER use this client in user-facing request handlers.
    """
    global _service_client
    if _service_client is None:
        settings = get_settings()
        _service_client = await acreate_client(
            settings.supabase_url,
            settings.supabase_service_role_key,
        )
        logger.info("Supabase service client initialised (RLS bypassed)")
    return _service_client


async def get_authed_client(jwt: str) -> AsyncClient:
    """Return a client scoped to a specific user JWT.

    This is the most restrictive client — it enforces RLS AND restricts
    the session to the authenticated user's identity.
    Use when you want maximum per-user isolation.
    """
    settings = get_settings()
    client = await acreate_client(
        settings.supabase_url,
        settings.supabase_anon_key,
    )
    await client.auth.set_session(access_token=jwt, refresh_token="")
    return client


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────


async def check_db_health() -> bool:
    """Verify Supabase connectivity. Used in /health endpoint."""
    try:
        client = await get_service_client()
        await client.table("health_check").select("id").limit(1).execute()
        return True
    except Exception as exc:
        logger.error("Supabase health check failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Shutdown
# ─────────────────────────────────────────────────────────────────────────────


async def close_clients() -> None:
    """Release Supabase clients on app shutdown."""
    global _anon_client, _service_client
    _anon_client = None
    _service_client = None
    logger.info("Supabase clients released")


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers (replaces Redis session cache)
# Session state lives in Supabase; hot path uses in-memory dict per WS conn.
# ─────────────────────────────────────────────────────────────────────────────


async def get_session_state(session_id: str) -> dict[str, Any] | None:
    """Retrieve session state from Supabase."""
    try:
        client = await get_service_client()
        result = (
            await client.table("sessions")
            .select("state_json")
            .eq("id", session_id)
            .single()
            .execute()
        )
        if result.data:
            return result.data.get("state_json") or {}
        return None
    except Exception as exc:
        logger.error("Failed to get session state [%s]: %s", session_id, exc)
        return None


async def set_session_state(session_id: str, state: dict[str, Any]) -> bool:
    """Persist session state to Supabase."""
    try:
        client = await get_service_client()
        await (
            client.table("sessions")
            .update({"state_json": state})
            .eq("id", session_id)
            .execute()
        )
        return True
    except Exception as exc:
        logger.error("Failed to set session state [%s]: %s", session_id, exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Semantic profile helpers (replaces Redis profile cache)
# ─────────────────────────────────────────────────────────────────────────────


async def get_user_profile(user_id: str) -> dict[str, Any] | None:
    """Fetch the user's semantic profile (extracted facts only, no raw text)."""
    try:
        client = await get_service_client()
        result = (
            await client.table("user_profiles")
            .select("*")
            .eq("user_id", user_id)
            .single()
            .execute()
        )
        return result.data
    except Exception as exc:
        logger.debug("No profile found for user %s: %s", user_id, exc)
        return None


async def upsert_user_profile(user_id: str, updates: dict[str, Any]) -> bool:
    """Create or update a user's semantic profile."""
    try:
        client = await get_service_client()
        await (
            client.table("user_profiles")
            .upsert({"user_id": user_id, **updates}, on_conflict="user_id")
            .execute()
        )
        return True
    except Exception as exc:
        logger.error("Failed to upsert profile [%s]: %s", user_id, exc)
        return False
