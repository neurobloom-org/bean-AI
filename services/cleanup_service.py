"""BEAN AI — Cleanup service.

Owns utility functions for purging stale data that lives outside the main
privacy_service scope (rate limits, ephemeral counters).

Consumed by background/session_cleanup.py → run_all_purges() and
run_transcript_purge_loop().

Exported functions:
    clean_expired_rate_limits() — delete stale rows from rate_limits table.

Note:
    The canonical `check_and_increment_rate_limit` logic lives in
    api/middleware/rate_limiter.py. This module only handles cleanup — it
    never reads or writes active rate-limit counters.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from services.supabase_client import get_service_client

logger = logging.getLogger(__name__)

# Rate-limit windows are at most 1 minute. A row whose window_start is
# older than _STALE_AFTER is guaranteed to be expired regardless of the
# configured window length, so it is safe to delete.
_STALE_AFTER_MINUTES: int = 2


async def clean_expired_rate_limits() -> int:
    """Delete rate_limit rows whose window has fully expired.

    Uses a 2-minute cutoff — double the maximum window length — to ensure
    no active window is ever deleted, even with minor clock skew across
    replicas.

    Returns:
        Number of rows deleted. Returns 0 on failure (never raises).
    """
    cutoff = datetime.now(UTC) - timedelta(minutes=_STALE_AFTER_MINUTES)

    try:
        client = await get_service_client()
        result = (
            await client.table("rate_limits")
            .delete()
            .lt("window_start", cutoff.isoformat())
            .execute()
        )
        count = len(result.data) if result.data else 0
        if count:
            logger.info("[Cleanup] Deleted %d stale rate_limit rows", count)
        return count

    except Exception as exc:
        logger.warning("[Cleanup] Failed to clean rate_limits: %s", exc)
        return 0