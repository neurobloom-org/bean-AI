"""BEAN AI — Emotion log purge background job.

Deletes emotion_events rows older than settings.emotion_retention_days
(default: 90 days). Runs every PURGE_INTERVAL_HOURS (6 hours).

This is the single owner of scheduled emotion_events deletion.
session_cleanup.py must NOT also delete from emotion_events on a schedule —
doing so causes redundant double-deletes. session_cleanup.py's
purge_expired_emotions() exists only for the admin run_all_purges() trigger.

Deletion strategy:
    Rows where detected_at < NOW() - retention_days are hard-deleted.
    This is an age-based cutoff complementary to the expires_at column
    (which is set to detected_at + 90 days by the DB default). In practice
    both strategies target the same rows; using detected_at is more explicit
    and independent of whether expires_at was correctly set at insert time.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Final

from services.supabase_client import get_service_client
from shared.config import get_settings

__all__ = [
    "purge_old_emotion_events",
    "run_emotion_purge_loop",
]

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# How often the purge loop runs.
PURGE_INTERVAL_HOURS: Final[int] = 6

# Timeout for the Supabase delete call.
_DB_TIMEOUT_SECONDS: Final[float] = 15.0


# ── Purge task ────────────────────────────────────────────────────────────────


async def purge_old_emotion_events() -> int:
    """Delete emotion_events rows older than the configured retention period.

    Uses detected_at (when the emotion was recorded) as the cutoff column
    rather than expires_at, so rows are deleted based on their actual age
    regardless of whether expires_at was set correctly at insert time.

    Returns:
        Number of rows deleted. Returns 0 on any failure (already logged).
    """
    settings = get_settings()
    retention_days = settings.emotion_retention_days
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()

    try:
        client = await get_service_client()

        result = await asyncio.wait_for(
            client.table("emotion_events").delete().lt("detected_at", cutoff).execute(),
            timeout=_DB_TIMEOUT_SECONDS,
        )

        count = len(result.data) if result.data else 0

        if count:
            logger.info(
                "[EmotionPurge] Deleted %d emotion events older than %d days "
                "(cutoff=%s)",
                count,
                retention_days,
                cutoff[:10],
            )
        else:
            logger.debug(
                "[EmotionPurge] No emotion events to purge (retention=%d days)",
                retention_days,
            )

        return count

    except asyncio.TimeoutError:
        logger.error(
            "[EmotionPurge] DB delete timed out after %.1fs", _DB_TIMEOUT_SECONDS
        )
        return 0

    except Exception as exc:
        logger.error("[EmotionPurge] Failed: %s", exc)
        return 0


# ── Background loop ───────────────────────────────────────────────────────────


async def run_emotion_purge_loop() -> None:
    """Run emotion purge every PURGE_INTERVAL_HOURS hours.

    This is the exclusive owner of scheduled emotion_events deletion.
    Exits cleanly on asyncio.CancelledError (FastAPI lifespan shutdown).
    All other exceptions are caught so one bad cycle never kills the loop.
    """
    interval_seconds = PURGE_INTERVAL_HOURS * 3600

    logger.info("[EmotionPurge] Loop started — interval=%dh", PURGE_INTERVAL_HOURS)

    while True:
        try:
            count = await purge_old_emotion_events()

            if count:
                logger.info("[EmotionPurge] Cycle complete — deleted %d rows", count)

            await asyncio.sleep(interval_seconds)

        except asyncio.CancelledError:
            logger.info("[EmotionPurge] Loop cancelled — shutting down cleanly")
            break

        except Exception as exc:
            logger.error(
                "[EmotionPurge] Loop error (will retry in %dh): %s",
                PURGE_INTERVAL_HOURS,
                exc,
            )
            await asyncio.sleep(interval_seconds)
