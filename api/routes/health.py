"""BEAN AI v1 — Health check routes."""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status
from fastapi.responses import JSONResponse

from services.supabase_client import check_db_health
from shared.config import Settings, get_settings
from shared.schemas import HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

DB_HEALTH_TIMEOUT_SECONDS = 2.0
NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}


async def _build_readiness_response(settings: Settings) -> HealthResponse:
    """Build a readiness response.

    Readiness includes downstream dependency checks like the database.
    """
    try:
        supabase_ok = await asyncio.wait_for(
            check_db_health(),
            timeout=DB_HEALTH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            "DB health check timed out after %.1f seconds",
            DB_HEALTH_TIMEOUT_SECONDS,
        )
        supabase_ok = False
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        supabase_ok = False

    return HealthResponse(
        status="ok" if supabase_ok else "degraded",
        version=settings.app_version,
        environment=settings.environment,
        supabase=supabase_ok,
        deepgram_configured=bool(settings.deepgram_api_key),
        elevenlabs_configured=bool(settings.elevenlabs_api_key),
        gemini_configured=bool(settings.google_api_key),
    )


@router.get("/") # type: ignore[untyped-decorator]
async def root_health_check(response: Response) -> dict[str, str]:
    """Liveness probe.

    This endpoint must stay lightweight and never hit the database.
    It only answers whether the Python app process is up.
    """
    response.headers.update(NO_CACHE_HEADERS)
    return {"status": "ok"}


@router.get("/health", response_model=HealthResponse) # type: ignore[untyped-decorator]
async def health_check(
    response: Response,
    settings: Annotated[Settings, Depends(get_settings)],
) -> HealthResponse | JSONResponse:
    """Readiness probe.

    This endpoint checks whether the app can still reach critical dependencies.
    """
    response.headers.update(NO_CACHE_HEADERS)
    health = await _build_readiness_response(settings)

    if health.status == "degraded":
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=health.model_dump(),
            headers=NO_CACHE_HEADERS,
        )

    return health
