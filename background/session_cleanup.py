"""BEAN AI — Session cleanup and data purge background jobs.

Two independent loops are exported for use in api/main.py:

    run_transcript_purge_loop()
        Runs every transcript_purge_interval_minutes (default: 60 min).
        Purges: expired transcripts, expired episodic memories,
                expired TTS cache entries, stale rate-limit records.
        Does NOT purge emotion_events — that is handled exclusively by
        background/emotion_purge.py to avoid double-deletes.

    run_session_cleanup_loop()
        Runs every session_cleanup_interval_hours (default: 6 h).
        Marks stale active sessions as expired; deletes old ended sessions.

One admin helper is also exported:

    run_all_purges()
        Runs every purge task once synchronously. Called by admin routes
        or management scripts that need an immediate full sweep.
        This DOES include emotion purge (via purge_expired_emotions) since
        the admin trigger should cover the full data surface.

Ownership of emotion_events:
    The scheduled deletion of emotion_events is owned exclusively by
    background/emotion_purge.py (run_emotion_purge_loop).
    purge_expired_emotions() remains here only for run_all_purges().
    Do NOT add it back to run_transcript_purge_loop.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from services.cleanup_service import clean_expired_rate_limits
from services.privacy_service import privacy_service
from services.supabase_client import get_service_client
from shared.config import get_settings

__all__ = [
    "run_transcript_purge_loop",
    "run_session_cleanup_loop",
    "run_all_purges",
    "purge_expired_transcripts",
    "purge_expired_emotions",
    "purge_expired_memories",
    "purge_expired_tts_cache",
    "cleanup_rate_limits",
    "cleanup_ended_sessions",
]

logger = logging.getLogger(__name__)

# Timeout for individual Supabase calls inside cleanup_ended_sessions.
_DB_TIMEOUT_SECONDS: float = 10.0


# ── Individual purge tasks ────────────────────────────────────────────────────


async def purge_expired_transcripts() -> int:
    """Delete session_transcripts rows whose expires_at has passed.

    Returns the number of rows deleted (0 on failure).
    """
    try:
        count = await privacy_service.purge_expired_transcripts()
        if count:
            logger.info("[Purge] Deleted %d expired transcript rows", count)
        return count or 0
    except Exception as exc:
        logger.error("[Purge] Transcript purge failed: %s", exc)
        return 0


async def purge_expired_emotions() -> int:
    """Delete emotion_events rows whose expires_at has passed.

    This function exists for run_all_purges() (admin trigger).
    Do NOT call this from run_transcript_purge_loop — emotion_purge.py
    owns the scheduled deletion.

    Returns the number of rows deleted (0 on failure).
    """
    try:
        count = await privacy_service.purge_expired_emotion_events()
        if count:
            logger.info("[Purge] Deleted %d expired emotion rows", count)
        return count or 0
    except Exception as exc:
        logger.error("[Purge] Emotion purge failed: %s", exc)
        return 0


async def purge_expired_memories() -> int:
    """Delete episodic_memories rows whose expires_at has passed.

    Returns the number of rows deleted (0 on failure).
    """
    try:
        count = await privacy_service.purge_expired_episodic_memories()
        if count:
            logger.info("[Purge] Deleted %d expired episodic memories", count)
        return count or 0
    except Exception as exc:
        logger.error("[Purge] Memory purge failed: %s", exc)
        return 0


async def purge_expired_tts_cache() -> int:
    """Delete tts_cache rows whose expires_at has passed.

    The tts_cache table has a 7-day TTL (set in 001_schema.sql).
    Nothing else clears it — without this the table grows unboundedly.

    Returns the number of rows deleted (0 on failure).
    """
    try:
        client = await get_service_client()
        now = datetime.now(UTC).isoformat()

        result = await asyncio.wait_for(
            client.table("tts_cache").delete().lt("expires_at", now).execute(),
            timeout=_DB_TIMEOUT_SECONDS,
        )

        count = len(result.data) if result.data else 0
        if count:
            logger.info("[Purge] Deleted %d expired TTS cache entries", count)
        return count

    except asyncio.TimeoutError:
        logger.error(
            "[Purge] TTS cache purge timed out after %.1fs", _DB_TIMEOUT_SECONDS
        )
        return 0
    except Exception as exc:
        logger.error("[Purge] TTS cache purge failed: %s", exc)
        return 0


async def cleanup_rate_limits() -> int:
    """Delete stale rate_limits rows older than 1 hour.

    Returns the number of rows deleted (0 on failure).
    """
    try:
        count = await clean_expired_rate_limits()
        if count:
            logger.info("[Purge] Deleted %d expired rate-limit rows", count)
        return count or 0
    except Exception as exc:
        logger.error("[Purge] Rate-limit cleanup failed: %s", exc)
        return 0


async def cleanup_ended_sessions() -> int:
    """Mark stale active sessions as expired and delete old ended sessions.

    Step 1: sessions with status='active' and started_at older than 24 hours
            are updated to status='expired'. These are sessions whose WebSocket
            connection was never cleanly closed (e.g. process restart, crash).

    Step 2: sessions with status in ('ended', 'expired') and ended_at older
            than session_retention_days are hard-deleted.

    Each step is independent — a failure in step 1 does not block step 2.

    Returns the total number of rows changed across both steps (0 on failure).
    """
    settings = get_settings()
    now = datetime.now(UTC)
    total_changed = 0

    # ── Step 1: expire stale active sessions ──────────────────────────────────
    try:
        client = await get_service_client()
        stale_threshold = (now - timedelta(hours=24)).isoformat()

        expire_result = await asyncio.wait_for(
            client.table("sessions")
            .update(
                {
                    "status": "expired",
                    "ended_at": now.isoformat(),
                }
            )
            .eq("status", "active")
            .lt("started_at", stale_threshold)
            .execute(),
            timeout=_DB_TIMEOUT_SECONDS,
        )

        expired_count = len(expire_result.data) if expire_result.data else 0
        total_changed += expired_count

        if expired_count:
            logger.info("[Cleanup] Marked %d stale sessions as expired", expired_count)

    except asyncio.TimeoutError:
        logger.error(
            "[Cleanup] Expire stale sessions timed out after %.1fs",
            _DB_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.error("[Cleanup] Failed to expire stale sessions: %s", exc)

    # ── Step 2: delete old ended/expired sessions ─────────────────────────────
    try:
        client = await get_service_client()
        retention_days = settings.session_retention_days
        delete_before = (now - timedelta(days=retention_days)).isoformat()

        delete_result = await asyncio.wait_for(
            client.table("sessions")
            .delete()
            .in_("status", ["ended", "expired"])
            .lt("ended_at", delete_before)
            .execute(),
            timeout=_DB_TIMEOUT_SECONDS,
        )

        deleted_count = len(delete_result.data) if delete_result.data else 0
        total_changed += deleted_count

        if deleted_count:
            logger.info(
                "[Cleanup] Deleted %d old ended/expired sessions (retention=%d days)",
                deleted_count,
                retention_days,
            )

    except asyncio.TimeoutError:
        logger.error(
            "[Cleanup] Delete old sessions timed out after %.1fs",
            _DB_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        logger.error("[Cleanup] Failed to delete old sessions: %s", exc)

    return total_changed


# ── Background loops ──────────────────────────────────────────────────────────


async def run_transcript_purge_loop() -> None:
    """Purge expired transcripts, memories, TTS cache, and rate limits.

    Runs every transcript_purge_interval_minutes (default: 60 min).

    Intentionally does NOT purge emotion_events — that is the exclusive
    responsibility of background/emotion_purge.py. Running both would
    cause redundant double-deletes on the emotion_events table.
    """
    settings = get_settings()
    interval = settings.transcript_purge_interval_minutes * 60

    logger.info(
        "[Purge] Transcript purge loop started — interval=%dm",
        settings.transcript_purge_interval_minutes,
    )

    while True:
        try:
            await purge_expired_transcripts()
            await purge_expired_memories()
            await purge_expired_tts_cache()
            await cleanup_rate_limits()
            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.info("[Purge] Transcript purge loop cancelled — shutting down")
            break

        except Exception as exc:
            logger.error("[Purge] Purge loop error (will retry): %s", exc)
            await asyncio.sleep(interval)


async def run_session_cleanup_loop() -> None:
    """Mark stale sessions as expired and delete old ended sessions.

    Runs every session_cleanup_interval_hours (default: 6 h).
    """
    settings = get_settings()
    interval = settings.session_cleanup_interval_hours * 3600

    logger.info(
        "[Cleanup] Session cleanup loop started — interval=%dh",
        settings.session_cleanup_interval_hours,
    )

    while True:
        try:
            await cleanup_ended_sessions()
            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.info("[Cleanup] Session cleanup loop cancelled — shutting down")
            break

        except Exception as exc:
            logger.error("[Cleanup] Session cleanup error (will retry): %s", exc)
            await asyncio.sleep(interval)


# ── Admin helper ──────────────────────────────────────────────────────────────


async def run_all_purges() -> dict[str, int]:
    """Run every purge and cleanup task once immediately.

    Intended for admin routes or management scripts that need a full sweep
    on demand. Unlike the scheduled loops, this DOES include emotion purge
    so the admin trigger covers the entire data surface.

    Returns a dict mapping task name → number of rows affected.
    """
    transcripts_deleted = await purge_expired_transcripts()
    emotions_deleted = await purge_expired_emotions()
    memories_deleted = await purge_expired_memories()
    tts_cache_deleted = await purge_expired_tts_cache()
    rate_limits_deleted = await cleanup_rate_limits()
    sessions_changed = await cleanup_ended_sessions()

    return {
        "transcripts_deleted": transcripts_deleted,
        "emotions_deleted": emotions_deleted,
        "memories_deleted": memories_deleted,
        "tts_cache_deleted": tts_cache_deleted,
        "rate_limits_deleted": rate_limits_deleted,
        "sessions_changed": sessions_changed,
    }