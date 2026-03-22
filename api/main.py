"""BEAN AI v1 — FastAPI application entry point.

Lifespan contract:
  Startup  — DB health check → background tasks → ready
  Shutdown — cancel tasks → close HTTP clients → close Supabase clients

Background jobs (all started as asyncio.Task):
  transcript-purge   — purge expired transcripts, memories, TTS cache, rate-limits
  session-cleanup    — expire/delete stale sessions
  reminder-check     — send SMS reminders for due tasks
  emotion-purge      — age-based deletion of emotion_events (owned here exclusively)

Internal routes:
  POST /internal/purge — admin-only full purge sweep; requires X-Internal-Key header.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.middleware.auth_middleware import SupabaseAuthMiddleware
from api.middleware.rate_limiter import RateLimiterMiddleware
from api.routes import alerts, auth, emotions, guardian, health, sessions, tasks
from api.websocket_handler import router as ws_router
from background.emotion_purge import run_emotion_purge_loop
from background.reminder_check import run_reminder_check_loop
from background.session_cleanup import run_all_purges, run_session_cleanup_loop, run_transcript_purge_loop
from services.supabase_client import check_db_health, close_clients
from shared.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Module-level task registry (populated in lifespan) ────────────────────────
_background_tasks: list[asyncio.Task] = []


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and graceful shutdown for BEAN AI."""
    settings = get_settings()
    logger.info(
        "BEAN AI v%s starting [env=%s]",
        settings.app_version,
        settings.environment,
    )

    # ── Startup ───────────────────────────────────────────────────────────────

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

    # Start background jobs.
    # FIX: Added run_emotion_purge_loop — it was missing from the original and
    # is the single owner of scheduled emotion_events deletion (per emotion_purge.py).
    _background_tasks.extend(
        [
            asyncio.create_task(
                run_transcript_purge_loop(), name="transcript-purge"
            ),
            asyncio.create_task(
                run_session_cleanup_loop(), name="session-cleanup"
            ),
            asyncio.create_task(
                run_reminder_check_loop(), name="reminder-check"
            ),
            asyncio.create_task(
                run_emotion_purge_loop(), name="emotion-purge"
            ),
        ]
    )

    logger.info(
        "✓ Background jobs started (%d tasks): %s",
        len(_background_tasks),
        ", ".join(t.get_name() for t in _background_tasks),
    )
    logger.info("✓ BEAN AI ready — listening on :%d", settings.port)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────

    logger.info("BEAN AI shutting down…")

    # Cancel all background tasks and wait for them to exit cleanly.
    for task in _background_tasks:
        task.cancel()
    if _background_tasks:
        await asyncio.gather(*_background_tasks, return_exceptions=True)
    logger.info("✓ Background tasks cancelled")

    # FIX: Close the shared httpx.AsyncClient used by auth routes for Google
    # OAuth token exchange.  Without this the process leaks an open connection
    # pool on every restart.
    try:
        from api.routes.auth import OAUTH_HTTP_CLIENT
        await OAUTH_HTTP_CLIENT.aclose()
        logger.info("✓ OAuth HTTP client closed")
    except Exception as exc:  # pragma: no cover
        logger.warning("OAuth HTTP client close failed (non-critical): %s", exc)

    await close_clients()
    logger.info("✓ Supabase clients closed")
    logger.info("✓ BEAN AI shutdown complete")


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "BEAN AI — Privacy-First Mental Health Companion Robot API. "
            "Supports ESP32-S3 Nano hardware via WebSocket real-time audio pipeline."
        ),
        # Disable interactive docs in production to reduce attack surface.
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # ── Middleware ─────────────────────────────────────────────────────────────
    # Starlette applies middleware bottom-up, so the outermost (last added) runs
    # first.  Desired order: CORS → Auth → RateLimiter → handler.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimiterMiddleware)
    app.add_middleware(SupabaseAuthMiddleware)

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(sessions.router)
    app.include_router(alerts.router)
    app.include_router(emotions.router)
    app.include_router(tasks.router)
    app.include_router(guardian.router)
    app.include_router(ws_router)

    # Internal admin routes (authenticated by shared secret header).
    _register_internal_routes(app)

    return app


def _register_internal_routes(app: FastAPI) -> None:
    """Register /internal/* admin endpoints.

    These endpoints are NOT exposed via the public API router and are
    guarded by an X-Internal-Key header check.  In production, Cloud Run
    ingress rules should additionally restrict access to internal IPs only.

    FIX: The original code had no authentication on /internal/purge, making
    it an unauthenticated DoS vector.  This version validates a shared secret.
    """
    from fastapi import APIRouter, Header, HTTPException

    internal = APIRouter(prefix="/internal", tags=["internal"])

    def _require_internal_key(
        x_internal_key: str | None = Header(default=None),
    ) -> None:
        """Validate the shared internal key sent by orchestration scripts."""
        settings = get_settings()
        expected = getattr(settings, "internal_api_key", None) or ""
        if not expected:
            # If no key is configured the endpoint is disabled in production.
            if settings.is_production:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Internal endpoints are not configured in production.",
                )
            return  # Allow unauthenticated access in non-production only.
        if not x_internal_key or x_internal_key != expected:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-Internal-Key header.",
            )

    @internal.post("/purge")
    async def trigger_purge(request: Request) -> JSONResponse:
        """Trigger an immediate full data purge sweep (admin only)."""
        _require_internal_key(
            request.headers.get("x-internal-key") or request.headers.get("X-Internal-Key")
        )
        try:
            results = await run_all_purges()
            return JSONResponse(content={"status": "ok", "results": results})
        except Exception as exc:
            logger.error("Manual purge failed: %s", exc)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"status": "error", "detail": str(exc)},
            )

    @internal.get("/health")
    async def internal_health() -> JSONResponse:
        """Liveness probe for internal orchestration systems."""
        return JSONResponse(content={"status": "ok", "tasks": len(_background_tasks)})

    app.include_router(internal)


# ─────────────────────────────────────────────────────────────────────────────
# Application singleton
# ─────────────────────────────────────────────────────────────────────────────

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
