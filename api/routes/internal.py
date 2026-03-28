"""Internal admin routes — protected by X-Internal-Key header.

These routes are NOT protected by Supabase JWT middleware. Instead they
require the INTERNAL_API_KEY header (verified by verify_internal_key dep).
The /internal/ path prefix is exempted from SupabaseAuthMiddleware in
api/middleware/auth_middleware.py (PUBLIC_PATH_PREFIXES).
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends

from api.routes._deps import verify_internal_key
from background.session_cleanup import run_all_purges
from services.supabase_client import check_db_health
from services.redis_service import get_redis

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/internal",
    tags=["internal"],
    dependencies=[Depends(verify_internal_key)],
)


@router.post("/purge")
async def trigger_purge() -> dict[str, Any]:
    """Trigger an immediate full data purge across all retention tables.

    Runs transcript, emotion, memory, TTS cache, rate limit, and session
    cleanup in sequence. Returns per-table deletion counts.
    """
    logger.info("[Internal] Manual purge triggered")
    result = await run_all_purges()
    logger.info("[Internal] Purge complete: %s", result)
    return {"status": "ok", "deleted": result}


@router.get("/health")
async def internal_health() -> dict[str, Any]:
    """Detailed health check including all backing services."""
    db_ok = await check_db_health()

    redis_ok = False
    try:
        r = await get_redis()
        await r.ping()
        redis_ok = True
    except Exception as exc:
        logger.warning("[Internal] Redis health check failed: %s", exc)

    return {
        "status": "healthy" if (db_ok and redis_ok) else "degraded",
        "supabase": db_ok,
        "redis": redis_ok,
    }
