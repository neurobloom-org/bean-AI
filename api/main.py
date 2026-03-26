"""BEAN AI v1 — FastAPI application entry point."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.middleware.auth_middleware import SupabaseAuthMiddleware
from api.middleware.rate_limiter import RateLimiterMiddleware
from api.websocket_handler import router as ws_router
from background.emotion_purge import run_emotion_purge_loop
from background.reminder_check import run_reminder_check_loop
from background.session_cleanup import (
    run_session_cleanup_loop,
    run_transcript_purge_loop,
)
from services.supabase_client import check_db_health, close_clients
from shared.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

_background_tasks: list[asyncio.Task] = []


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    logger.info(
        "BEAN AI v%s starting [env=%s]",
        settings.app_version,
        settings.environment,
    )

    db_healthy = await check_db_health()
    if not db_healthy:
        logger.critical(
            "Supabase health check FAILED on startup. "
            "Check SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
        )
        if settings.is_production:
            raise RuntimeError(
                "Supabase connectivity is required in production — aborting startup."
            )
        logger.warning("Continuing in degraded mode (non-production environment).")
    else:
        logger.info("✓ Supabase connected")

    _background_tasks.extend([
        asyncio.create_task(run_transcript_purge_loop(), name="transcript-purge"),
        asyncio.create_task(run_session_cleanup_loop(), name="session-cleanup"),
        asyncio.create_task(run_reminder_check_loop(), name="reminder-check"),
        asyncio.create_task(run_emotion_purge_loop(), name="emotion-purge"),
    ])
    logger.info(
        "✓ Background jobs started (%d tasks): %s",
        len(_background_tasks),
        ", ".join(t.get_name() for t in _background_tasks),
    )
    logger.info("✓ BEAN AI ready — listening on :%d", settings.port)

    try:
        yield
    finally:
        logger.info("BEAN AI shutting down…")
        for task in _background_tasks:
            task.cancel()
        if _background_tasks:
            await asyncio.gather(*_background_tasks, return_exceptions=True)
        logger.info("✓ Background tasks cancelled")
        await close_clients()
        logger.info("✓ Supabase clients closed")
        _background_tasks.clear()
        logger.info("✓ BEAN AI shutdown complete")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "BEAN AI — Privacy-First Mental Health Companion Robot API. "
            "Supports ESP32-S3 Nano hardware via WebSocket real-time audio pipeline."
        ),
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Health & root routes ──────────────────────────────────────────────────
    @app.get("/")
    async def root() -> dict:
        return {
            "status": "ok",
            "service": settings.app_name,
            "version": settings.app_version,
        }

    @app.get("/health")
    @app.get("/api/v1/health")
    async def health() -> JSONResponse:
        db_ok = await check_db_health()
        return JSONResponse(
            status_code=200 if db_ok else 503,
            content={
                "status": "healthy" if db_ok else "degraded",
                "version": settings.app_version,
                "environment": settings.environment,
                "supabase": db_ok,
                "deepgram_configured": bool(settings.deepgram_api_key),
                "elevenlabs_configured": bool(settings.elevenlabs_api_key),
                "gemini_configured": bool(settings.google_api_key),
            },
        )
    # ─────────────────────────────────────────────────────────────────────────

    app.include_router(ws_router)

    # ── Middleware stack ──────────────────────────────────────────────────────
    # FIX: In FastAPI/Starlette, add_middleware() prepends to the internal list
    # and the middleware stack is built with reversed(), so the LAST call to
    # add_middleware() becomes the OUTERMOST middleware (first to handle every
    # incoming request and last to handle every outgoing response).
    #
    # Correct execution order for a request:
    #   CORSMiddleware → SupabaseAuthMiddleware → RateLimiterMiddleware → route
    #
    # Why CORS must be outermost:
    #   - CORS headers must be present on ALL responses, including 401s from
    #     the auth middleware. If CORSMiddleware is innermost it only runs after
    #     auth has already returned, so the browser sees a 401 without CORS
    #     headers and reports a CORS error instead of the actual auth failure.
    #   - OPTIONS preflight requests must be handled by CORS before they ever
    #     reach auth or rate-limiting logic.
    #
    # Why Auth before RateLimit:
    #   - The rate limiter uses request.state.user_id (set by auth middleware)
    #     for per-user rate limiting. Auth must run first so user_id is available.
    #
    # add_middleware call order (last = outermost):
    #   1. RateLimiterMiddleware     ← added first → innermost
    #   2. SupabaseAuthMiddleware    ← added second → middle
    #   3. CORSMiddleware            ← added last  → outermost
    app.add_middleware(RateLimiterMiddleware)
    app.add_middleware(SupabaseAuthMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )