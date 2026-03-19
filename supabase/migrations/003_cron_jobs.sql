-- ============================================================================
-- BEAN AI v5 — Supabase pg_cron Scheduled Jobs Migration 003
-- DB-level data purge (more reliable than app-level background tasks)
-- ============================================================================
-- Requires: Supabase Pro plan (pg_cron available by default)
-- If on free plan: rely on background/session_cleanup.py app-level jobs instead
-- ============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- Transcript purge — every hour on the hour
-- Core privacy enforcement: removes 24h-expired conversation text
-- ─────────────────────────────────────────────────────────────────────────────
-- ENABLE CRON EXTENSION

CREATE EXTENSION IF NOT EXISTS pg_cron;
GRANT USAGE ON SCHEMA cron TO postgres;

SELECT cron.schedule(
    'bean-purge-transcripts',
    '0 * * * *',
    $$
        DELETE FROM public.session_transcripts
        WHERE expires_at < NOW();
    $$
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Emotion event purge — daily at 02:00 UTC
-- Removes emotion records older than 90 days
-- ─────────────────────────────────────────────────────────────────────────────
SELECT cron.schedule(
    'bean-purge-emotions',
    '0 2 * * *',
    $$
        DELETE FROM public.emotion_events
        WHERE expires_at < NOW();
    $$
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Episodic memory expiry — daily at 03:00 UTC
-- Only deletes memories with an explicit expires_at set
-- ─────────────────────────────────────────────────────────────────────────────
SELECT cron.schedule(
    'bean-purge-episodic',
    '0 3 * * *',
    $$
        DELETE FROM public.episodic_memories
        WHERE expires_at IS NOT NULL
          AND expires_at < NOW();
    $$
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Stale session cleanup — every 6 hours
-- Marks sessions as 'expired' if they've been 'active' for > 24h
-- (handles ESP32 hard resets / disconnects without clean session end)
-- ─────────────────────────────────────────────────────────────────────────────
SELECT cron.schedule(
    'bean-cleanup-sessions',
    '0 */6 * * *',
    $$
        UPDATE public.sessions
        SET status = 'expired',
            ended_at = NOW()
        WHERE status = 'active'
          AND started_at < NOW() - INTERVAL '24 hours';
    $$
);

-- ─────────────────────────────────────────────────────────────────────────────
-- Rate limit cleanup — every hour at :30
-- Removes stale rate limit windows (older than 1 hour)
-- ─────────────────────────────────────────────────────────────────────────────
SELECT cron.schedule(
    'bean-cleanup-rate-limits',
    '30 * * * *',
    $$
        DELETE FROM public.rate_limits
        WHERE window_start < NOW() - INTERVAL '1 hour';
    $$
);


-- ─────────────────────────────────────────────────────────────────────────────
-- Verify scheduled jobs
-- ─────────────────────────────────────────────────────────────────────────────
-- Run this to confirm all jobs are registered:
-- SELECT jobid, jobname, schedule, command FROM cron.job WHERE jobname LIKE 'bean-%';