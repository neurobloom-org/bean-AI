async def run_transcript_purge_loop() -> None:
    """Run transcript purge every N minutes (configured via settings)."""
    settings = get_settings()
    interval = settings.transcript_purge_interval_minutes * 60
    logger.info(
        "[Purge] Transcript purge loop started — interval=%dm",
        settings.transcript_purge_interval_minutes,
    )
    while True:
        try:
            await purge_expired_transcripts()
            await purge_expired_emotions()
            await purge_expired_memories()
            await cleanup_rate_limits()
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("[Purge] Transcript purge loop cancelled — shutting down")
            break
        except Exception as exc:
            logger.error("[Purge] Purge loop error (will retry): %s", exc)
            await asyncio.sleep(interval)


async def run_session_cleanup_loop() -> None:
    """Run session cleanup every N hours (configured via settings)."""
    settings = get_settings()
    interval = settings.session_cleanup_interval_hours * 3600
    logger.info(
        "[Cleanup] Session cleanup loop started — interval=%dh",
        settings.session_cleanup_interval_hours,
    )
    while True:
        try:
            await cleanup_ended_sessions()
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("[Cleanup] Session cleanup loop cancelled — shutting down")
            break
        except Exception as exc:
            logger.error("[Cleanup] Session cleanup error (will retry): %s", exc)
            await asyncio.sleep(interval)