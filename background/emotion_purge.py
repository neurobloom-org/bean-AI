"""BEAN AI — Emotion log purge background job.

Deletes emotion_events older than the retention period.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from shared.config import get_settings

logger = logging.getLogger(__name__)


async def purge_old_emotion_events() -> int:
    """Delete emotion_events older than retention period."""
    from services.supabase_client import get_service_client

    settings = get_settings()
    retention_days = settings.emotion_retention_days
    cutoff = (datetime.now(UTC) - timedelta(days=retention_days)).isoformat()

    try:
        client = await get_service_client()
        result = (
            await client.table("emotion_events")
            .delete()
            .lt("detected_at", cutoff)
            .execute()
        )

        count = len(result.data) if result.data else 0
        if count:
            logger.info(
                "[EmotionPurge] Deleted %d emotion events older than %d days",
                count,
                retention_days,
            )
        return count
    except Exception as exc:
        logger.error("[EmotionPurge] Failed: %s", exc)
        return 0


async def run_emotion_purge_loop() -> None:
    """Run emotion purge on the same interval as transcript purge."""
    settings = get_settings()
    interval = settings.transcript_purge_interval_minutes * 60
    logger.info("[EmotionPurge] Loop started — interval=%dm", settings.transcript_purge_interval_minutes)

    while True:
        try:
            await purge_old_emotion_events()
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("[EmotionPurge] Loop cancelled")
            break
        except Exception as exc:
            logger.error("[EmotionPurge] Loop error: %s", exc)
            await asyncio.sleep(interval)