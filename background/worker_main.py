"""BEAN AI — Background worker entry point for Render Worker service.

Runs all background loops:
  - Reminder check (every 60s) — slips tasks into active sessions
  - Transcript purge (every 60min)
  - Emotion purge (daily)
  - Session cleanup (every 6h)

Handles SIGTERM (sent by Render before SIGKILL) for clean shutdown.
"""

import asyncio
import logging
import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("worker_main")

_tasks: list[asyncio.Task] = []


def _handle_sigterm() -> None:
    logger.info("SIGTERM received — cancelling background tasks")
    for task in _tasks:
        task.cancel()


async def main() -> None:
    from background.reminder_check import run_reminder_check_loop
    from background.session_cleanup import run_transcript_purge_loop, run_session_cleanup_loop
    from background.emotion_purge import run_emotion_purge_loop

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, _handle_sigterm)

    logger.info("Background worker starting...")

    _tasks.extend([
        asyncio.create_task(run_reminder_check_loop(), name="reminder-check"),
        asyncio.create_task(run_transcript_purge_loop(), name="transcript-purge"),
        asyncio.create_task(run_session_cleanup_loop(), name="session-cleanup"),
        asyncio.create_task(run_emotion_purge_loop(), name="emotion-purge"),
    ])

    logger.info(
        "Background loops started: %s",
        ", ".join(t.get_name() for t in _tasks),
    )

    try:
        await asyncio.gather(*_tasks, return_exceptions=True)
    finally:
        logger.info("Background worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
