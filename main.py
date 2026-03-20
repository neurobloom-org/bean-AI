import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import (
    alerts,
    auth,
    emotions,
    guardian,
    health,
    sessions,
    tasks,
)

from background_jobs.reminder_check import run_reminder_loop
from background_jobs.episodic_embedder import run_episodic_embedder_loop
from background_jobs.session_cleanup import (
    run_transcript_purge_loop,
    run_session_cleanup_loop,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    tasks_list = [
        asyncio.create_task(run_reminder_loop()),
        asyncio.create_task(run_episodic_embedder_loop()),
        asyncio.create_task(run_transcript_purge_loop()),
        asyncio.create_task(run_session_cleanup_loop()),
    ]

    logger.info("Background jobs started")

    try:
        yield
    finally:
        for task in tasks_list:
            task.cancel()

        await asyncio.gather(*tasks_list, return_exceptions=True)
        logger.info("Background jobs stopped")


app = FastAPI(lifespan=lifespan)


# register route modules
app.include_router(alerts.router)
app.include_router(auth.router)
app.include_router(emotions.router)
app.include_router(guardian.router)
app.include_router(health.router)
app.include_router(sessions.router)
app.include_router(tasks.router)