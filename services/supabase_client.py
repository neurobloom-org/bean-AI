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

from supabase import AsyncClient, acreate_client  # type: ignore[attr-defined]

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
        await client.table("sessions").select("id").limit(1).execute()
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
    for client, label in (
        (_anon_client, "anon"),
        (_service_client, "service"),
    ):
        if client is not None:
            # supabase-py 2.x AsyncClient does not expose a public aclose()
            # method. Calling it raises AttributeError and prevents clean
            # shutdown. Close the underlying httpx transport instead.
            try:
                if hasattr(client, "postgrest") and hasattr(client.postgrest, "aclose"):
                    await client.postgrest.aclose()
            except Exception as exc:
                logger.warning("Supabase %s client close error: %s", label, exc)
    _anon_client = None
    _service_client = None
    logger.info("Supabase clients closed")


# ─────────────────────────────────────────────────────────────────────────────
# Session state helpers
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
# Semantic profile helpers
# ─────────────────────────────────────────────────────────────────────────────


async def get_user_profile(user_id: str) -> dict[str, Any] | None:
    """Fetch the user's semantic profile (extracted facts only, no raw text).

    FIX: Changed .single() to .maybe_single().

    .single() raises PostgREST406Error ("JSON object requested, multiple
    (or no) rows returned") when no row exists — which happens for every
    new user who hasn't had a profile built yet. Although the exception is
    caught, it pollutes logs and relies on exception control flow for a
    completely normal case (new user = no profile yet).

    .maybe_single() returns None cleanly when zero rows match, which is
    the correct semantic for "profile may or may not exist".
    """
    try:
        client = await get_service_client()
        result = (
            await client.table("user_profiles")
            .select("*")
            .eq("user_id", user_id)
            .maybe_single()  # FIX: was .single() — raises when no profile exists
            .execute()
        )
        return dict(result.data) if result and result.data else None
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