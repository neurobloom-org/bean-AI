"""BEAN AI v1 — FastAPI application entry point."""

from __future__ import annotations

import asyncio
import json
import logging
import logging.config
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.middleware.auth_middleware import SupabaseAuthMiddleware
from api.middleware.rate_limiter import RateLimiterMiddleware
from api.websocket_handler import router as ws_router
from api.routes.auth import router as auth_router
from api.routes.sessions import router as sessions_router
from api.routes.tasks import router as tasks_router
from api.routes.alerts import router as alerts_router
from api.routes.emotions import router as emotions_router
from api.routes.guardian import router as guardian_router
from api.routes.internal import router as internal_router
from background.emotion_purge import run_emotion_purge_loop
from background.reminder_check import run_reminder_check_loop
from background.session_cleanup import (
    run_session_cleanup_loop,
    run_transcript_purge_loop,
)
from services.supabase_client import check_db_health, close_clients
from services.redis_service import get_redis, close_redis
from shared.config import get_settings
from shared.exceptions import BEANError

def _configure_logging() -> None:
    """Use JSON logging in production (parseable by Render log viewer),
    plain text in development."""
    _env = os.getenv("ENVIRONMENT", "development")
    if _env == "production":
        class _JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                return json.dumps({
                    "severity": record.levelname,
                    "message": record.getMessage(),
                    "logger": record.name,
                    "time": self.formatTime(record),
                    **({"exc_info": self.formatException(record.exc_info)} if record.exc_info else {}),
                })

        handler = logging.StreamHandler()
        handler.setFormatter(_JsonFormatter())
        logging.root.handlers = [handler]
        logging.root.setLevel(logging.INFO)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        )


_configure_logging()
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

    def _on_task_done(task: asyncio.Task) -> None:  # type: ignore[type-arg]
        if not task.cancelled():
            exc = task.exception()
            if exc is not None:
                logger.critical(
                    "Background task %r died unexpectedly: %s",
                    task.get_name(),
                    exc,
                    exc_info=exc,
                )

    run_workers = os.getenv("RUN_BACKGROUND_WORKERS", "true").lower() == "true"
    if run_workers:
        _background_tasks.extend([
            asyncio.create_task(run_transcript_purge_loop(), name="transcript-purge"),
            asyncio.create_task(run_session_cleanup_loop(), name="session-cleanup"),
            asyncio.create_task(run_reminder_check_loop(), name="reminder-check"),
            asyncio.create_task(run_emotion_purge_loop(), name="emotion-purge"),
        ])
        for task in _background_tasks:
            task.add_done_callback(_on_task_done)
        logger.info(
            "✓ Background jobs started (%d tasks): %s",
            len(_background_tasks),
            ", ".join(t.get_name() for t in _background_tasks),
        )
    else:
        logger.info("Background workers disabled (RUN_BACKGROUND_WORKERS=false)")

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
        await close_redis()
        logger.info("✓ Redis client closed")
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

    # ── Global exception handlers ─────────────────────────────────────────────
    @app.exception_handler(BEANError)
    async def bean_error_handler(request: Request, exc: BEANError) -> JSONResponse:
        logger.error("BEANError on %s %s: %s", request.method, request.url.path, exc)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

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

        redis_ok = False
        try:
            r = await get_redis()
            await r.ping()
            redis_ok = True
        except Exception as exc:
            logger.warning("Redis health check failed: %s", exc)

        overall = db_ok  # Redis degraded does not block service health
        return JSONResponse(
            status_code=200 if overall else 503,
            content={
                "status": "healthy" if overall else "degraded",
                "version": settings.app_version,
                "environment": settings.environment,
                "supabase": db_ok,
                "redis": redis_ok,
                "deepgram_configured": bool(settings.deepgram_api_key),
                "elevenlabs_configured": bool(settings.elevenlabs_api_key),
                "gemini_configured": bool(settings.google_api_key),
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(ws_router)
    app.include_router(auth_router)
    app.include_router(sessions_router)
    app.include_router(tasks_router)
    app.include_router(alerts_router)
    app.include_router(emotions_router)
    app.include_router(guardian_router)
    app.include_router(internal_router)

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

    # In production, use explicit method/header allowlists.
    # Wildcard allow_headers is rejected by some browsers when
    # allow_credentials=True (CORS spec forbids * + credentials).
    if settings.is_production:
        _cors_methods = ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
        _cors_headers = ["Authorization", "Content-Type", "X-Request-ID", "X-Internal-Key"]
    else:
        _cors_methods = ["*"]
        _cors_headers = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=_cors_methods,
        allow_headers=_cors_headers,
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
