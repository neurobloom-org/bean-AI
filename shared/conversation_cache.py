"""BEAN AI — 10-turn conversation cache.

Hot layer: Redis (fast, session-scoped, auto-expires after 24h).
Cold layer: DB session_transcripts (backup, persisted elsewhere).

Each turn is stored as {"role": "user"|"bean", "text": str, "ts": iso}.

FIX: The DB warm-up fallback in get_history() mapped r["speaker"] directly
as the role. The session_transcripts table stores speaker values of "user"
or "assistant", but the cache format and all consumers (casual_chat,
therapeutic_convo, task_agent etc.) use "user" and "bean". On a Redis
miss after reconnect, the warm-up would populate the cache with role
"assistant" instead of "bean". While most consumers use
`t['role'] == 'user' else 'BEAN'` which tolerates "assistant", the
inconsistency causes confusion in explicit role checks and future-proofs
the cache contract. Fixed: normalize "assistant" → "bean" during warm-up.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TypedDict

from services.redis_service import redis_delete, redis_get, redis_set
from services.supabase_client import get_service_client

logger = logging.getLogger(__name__)

HISTORY_SIZE = 10
_TTL = 86400  # 24 hours — matches transcript retention

# Canonical role values used throughout the codebase.
_ROLE_USER = "user"
_ROLE_BEAN = "bean"

# Speaker values stored in session_transcripts table.
_SPEAKER_USER = "user"
_SPEAKER_ASSISTANT = "assistant"


class Turn(TypedDict):
    role: str   # "user" or "bean"
    text: str
    ts: str     # ISO timestamp


def _speaker_to_role(speaker: str) -> str:
    """Normalise a session_transcripts.speaker value to the cache role format.

    session_transcripts uses: "user" | "assistant"
    Cache format uses:        "user" | "bean"

    The mapping is: "user" → "user", anything else (e.g. "assistant") → "bean".
    This ensures the cache contract is consistent regardless of whether the
    data came from Redis (where roles are already "bean") or the DB fallback.
    """
    return _ROLE_USER if speaker == _SPEAKER_USER else _ROLE_BEAN


async def append_turn(session_id: str, role: str, text: str) -> None:
    """Append one turn to the Redis cache only.

    DB persistence is handled elsewhere by privacy_service.store_transcript_turn()
    to avoid duplicate session_transcripts rows.
    """
    key = f"convo:{session_id}"
    history: list[Turn] = await redis_get(key) or []

    turn: Turn = {
        "role": role,
        "text": text,
        "ts": datetime.now(UTC).isoformat(),
    }
    history.append(turn)

    # Keep only the last HISTORY_SIZE turns in Redis
    if len(history) > HISTORY_SIZE:
        history = history[-HISTORY_SIZE:]

    await redis_set(key, history, ttl_seconds=_TTL)


async def get_history(session_id: str) -> list[Turn]:
    """Return up to 10 recent turns. Falls back to DB if Redis is cold."""
    key = f"convo:{session_id}"
    cached: list[Turn] | None = await redis_get(key)
    if cached:
        return cached[-HISTORY_SIZE:]

    # Redis miss — load from DB and warm the cache.
    # FIX: Normalize speaker → role during warm-up so the cache always
    # contains "user"/"bean" regardless of the DB's "user"/"assistant" values.
    try:
        client = await get_service_client()
        result = (
            await client.table("session_transcripts")
            .select("speaker, text, created_at")
            .eq("session_id", session_id)
            .order("created_at", desc=False)
            .limit(HISTORY_SIZE)
            .execute()
        )
        rows = result.data or []
        turns: list[Turn] = [
            {
                "role": _speaker_to_role(r["speaker"]),  # FIX: was r["speaker"] directly
                "text": r["text"],
                "ts": r["created_at"],
            }
            for r in rows
        ]
        if turns:
            await redis_set(key, turns, ttl_seconds=_TTL)
        return turns
    except Exception as exc:
        logger.warning("DB fallback for history failed [session=%s]: %s", session_id[:8], exc)
        return []


async def clear_history(session_id: str) -> None:
    """Delete the Redis cache for a session (called on session end)."""
    await redis_delete(f"convo:{session_id}")