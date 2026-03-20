import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

# Import API route modules
from api.routes import (
    alerts,
    auth,
    emotions,
    guardian,
    health,
    sessions,
    tasks,
)

# Import background job loops
from background.reminder_check import run_reminder_check_loop
from background.episodic_embedder import run_episodic_embedder_loop
from background.session_cleanup import (
    run_transcript_purge_loop,
    run_session_cleanup_loop,
)
from background.emotion_purge import run_emotion_purge_loop

# Initialize logger for lifecycle tracking
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager.

    Runs when the application starts and stops.

    Responsibilities:
    - Start all background job loops at server startup
    - Cancel them safely when the server shuts down
    """

    # Create asyncio tasks for each background loop
    tasks_list = [
        asyncio.create_task(
            run_transcript_purge_loop(),
            name="transcript-purge-loop",
        ),
        asyncio.create_task(
            run_session_cleanup_loop(),
            name="session-cleanup-loop",
        ),
        asyncio.create_task(
            run_emotion_purge_loop(),
            name="emotion-purge-loop",
        ),
        asyncio.create_task(
            run_episodic_embedder_loop(),
            name="episodic-embedder-loop",
        ),
        asyncio.create_task(
            run_reminder_check_loop(),
            name="reminder-check-loop",
        ),
    ]

    logger.info("Background loops started")

    try:
        # Application runs here
        yield

    finally:
        # Cancel all background jobs during shutdown
        for task in tasks_list:
            task.cancel()

        # Wait for tasks to finish cancellation gracefully
        await asyncio.gather(*tasks_list, return_exceptions=True)

        logger.info("Background loops stopped")


# Create FastAPI application instance with lifespan manager
app = FastAPI(lifespan=lifespan)


# Register API route modules
# These attach route handlers to the FastAPI application
app.include_router(alerts.router)
app.include_router(auth.router)
app.include_router(emotions.router)
app.include_router(guardian.router)
app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(tasks.router)