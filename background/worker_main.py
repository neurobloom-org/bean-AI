"""BEAN AI — Background worker entry point for Render Worker service.

Runs all background loops:
  - Reminder check (every 60s) — slips tasks into active sessions
  - Transcript purge (every 60min)
  - Emotion purge (daily)
  - Session cleanup (every 6h)
"""

import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("worker_main")


async def main() -> None:
    from background.reminder_check import run_reminder_check_loop
    from background.session_cleanup import run_transcript_purge_loop, run_session_cleanup_loop
    from background.emotion_purge import run_emotion_purge_loop

    logger.info("Background worker starting...")

    await asyncio.gather(
        run_reminder_check_loop(),
        run_transcript_purge_loop(),
        run_session_cleanup_loop(),
        run_emotion_purge_loop(),
    )


if __name__ == "__main__":
    asyncio.run(main())