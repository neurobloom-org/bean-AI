import asyncio
import logging
from datetime import UTC, datetime, timedelta

from api.middleware.rate_limiter import clean_expired_rate_limits
from services.privacy_service import privacy_service
from services.supabase_client import get_service_client
from shared.config import get_settings

logger = logging.getLogger(__name__)


async def purge_expired_transcripts() -> int:
    """Delete expired transcript rows."""
    try:
        count = await privacy_service.purge_expired_transcripts()
        if count:
            logger.info("[Purge] Deleted %d expired transcript rows", count)
        return count or 0
    except Exception as exc:
        logger.error("[Purge] Transcript purge failed: %s", exc)
        return 0


async def purge_expired_emotions() -> int:
    """Delete expired emotion rows."""
    try:
        count = await privacy_service.purge_expired_emotion_events()
        if count:
            logger.info("[Purge] Deleted %d expired emotion rows", count)
        return count or 0
    except Exception as exc:
        logger.error("[Purge] Emotion purge failed: %s", exc)
        return 0


async def purge_expired_memories() -> int:
    """Delete expired episodic memories."""
    try:
        count = await privacy_service.purge_expired_episodic_memories()
        if count:
            logger.info("[Purge] Deleted %d expired episodic memories", count)
        return count or 0
    except Exception as exc:
        logger.error("[Purge] Memory purge failed: %s", exc)
        return 0


async def cleanup_rate_limits() -> int:
    """Delete expired rate-limit records."""
    try:
        count = await clean_expired_rate_limits()
        if count:
            logger.info("[Purge] Deleted %d expired rate limit rows", count)
        return count or 0
    except Exception as exc:
        logger.error("[Purge] Rate-limit cleanup failed: %s", exc)
        return 0


async def cleanup_ended_sessions() -> int:
    """Mark stale active sessions as expired and delete old ended sessions."""
    settings = get_settings()
    client = await get_service_client()
    now = datetime.now(UTC)
    total_changed = 0

    try:
        stale_threshold = (now - timedelta(hours=24)).isoformat()

        expire_result = (
            await client.table("sessions")
            .update(
                {
                    "status": "expired",
                    "ended_at": now.isoformat(),
                }
            )
            .eq("status", "active")
            .lt("started_at", stale_threshold)
            .execute()
        )

        expired_count = len(expire_result.data) if expire_result.data else 0
        total_changed += expired_count

        if expired_count:
            logger.info("[Cleanup] Marked %d stale sessions as expired", expired_count)

    except Exception as exc:
        logger.error("[Cleanup] Failed to expire stale sessions: %s", exc)

    try:
        retention_days = settings.session_retention_days
        delete_before = (now - timedelta(days=retention_days)).isoformat()

        delete_result = (
            await client.table("sessions")
            .delete()
            .in_("status", ["ended", "expired"])
            .lt("ended_at", delete_before)
            .execute()
        )

        deleted_count = len(delete_result.data) if delete_result.data else 0
        total_changed += deleted_count

        if deleted_count:
            logger.info(
                "[Cleanup] Deleted %d old ended/expired sessions", deleted_count
            )

    except Exception as exc:
        logger.error("[Cleanup] Failed deleting old sessions: %s", exc)

    return total_changed


async def run_transcript_purge_loop() -> None:
    """Run transcript purge every N minutes (configured via settings)."""
    settings = get_settings()
    interval = settings.transcript_purge_interval_minutes * 60

    logger.info(
        "[Purge] Transcript purge loop started — interval=%dm",
        settings.transcript_purge_interval_minutes,
    )

    while True:
        try:
            await purge_expired_transcripts()
            await purge_expired_emotions()
            await purge_expired_memories()
            await cleanup_rate_limits()
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("[Purge] Transcript purge loop cancelled — shutting down")
            break
        except Exception as exc:
            logger.error("[Purge] Purge loop error (will retry): %s", exc)
            await asyncio.sleep(interval)


async def run_session_cleanup_loop() -> None:
    """Run session cleanup every N hours (configured via settings)."""
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


async def run_all_purges() -> dict[str, int]:
    """Run all purge and cleanup tasks once for manual admin trigger."""
    transcripts_deleted = await purge_expired_transcripts()
    emotion_deleted = await purge_expired_emotions()
    memories_deleted = await purge_expired_memories()
    sessions_expired = await cleanup_ended_sessions()
    rate_limits_deleted = await cleanup_rate_limits()

    return {
        "transcripts_deleted": transcripts_deleted,
        "emotion_deleted": emotion_deleted,
        "memories_deleted": memories_deleted,
        "sessions_expired": sessions_expired,
        "rate_limits_deleted": rate_limits_deleted,
    }
