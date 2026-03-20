-- ============================================================================
-- BEAN AI v1 — Supabase pg_cron Scheduled Jobs Migration 003
-- DB-level data purge (more reliable than app-level background tasks)
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS pg_cron;
GRANT USAGE ON SCHEMA cron TO postgres;

-- Remove existing jobs if rerun (safe to apply multiple times)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'bean-purge-transcripts') THEN
        PERFORM cron.unschedule('bean-purge-transcripts');
    END IF;
    IF EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'bean-purge-emotions') THEN
        PERFORM cron.unschedule('bean-purge-emotions');
    END IF;
    IF EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'bean-purge-episodic') THEN
        PERFORM cron.unschedule('bean-purge-episodic');
    END IF;
    IF EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'bean-cleanup-sessions') THEN
        PERFORM cron.unschedule('bean-cleanup-sessions');
    END IF;
    IF EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'bean-cleanup-rate-limits') THEN
        PERFORM cron.unschedule('bean-cleanup-rate-limits');
    END IF;
    IF EXISTS (SELECT 1 FROM cron.job WHERE jobname = 'bean-purge-tts-cache') THEN
        PERFORM cron.unschedule('bean-purge-tts-cache');
    END IF;
END $$;


-- ─────────────────────────────────────────────────────────────────────────────
-- Transcript purge — every hour on the hour
-- Core privacy enforcement: removes 24h-expired conversation text
-- ─────────────────────────────────────────────────────────────────────────────
SELECT cron.schedule(
    'bean-purge-transcripts',
    '0 * * * *',
    $$
        DELETE FROM public.session_transcripts
        WHERE expires_at < NOW();
    $$
);


-- ─────────────────────────────────────────────────────────────────────────────
-- Emotion events purge — daily at 2am UTC
-- Privacy: emotion data must not persist indefinitely (90-day TTL)
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
-- Episodic memory purge — daily at 3am UTC
-- Only deletes rows where expires_at is explicitly set (365-day TTL)
-- Rows with expires_at IS NULL are kept indefinitely
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
-- Session expiry — every 6 hours
-- FIX: also sets duration_seconds so guardian dashboard stats are not NULL
--      on force-expired sessions
-- ─────────────────────────────────────────────────────────────────────────────
SELECT cron.schedule(
    'bean-cleanup-sessions',
    '0 */6 * * *',
    $$
        UPDATE public.sessions
        SET status           = 'expired',
            ended_at         = NOW(),
            duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))::INTEGER
        WHERE status = 'active'
          AND started_at < NOW() - INTERVAL '24 hours';
    $$
);


-- ─────────────────────────────────────────────────────────────────────────────
-- Rate limit record cleanup — every hour at :30
-- Removes stale counters older than 1 hour to keep the table small
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
-- TTS cache purge — daily at 4am UTC
-- FIX: tts_cache has a 7-day expires_at TTL but had no cleanup job
-- ─────────────────────────────────────────────────────────────────────────────
SELECT cron.schedule(
    'bean-purge-tts-cache',
    '0 4 * * *',
    $$
        DELETE FROM public.tts_cache
        WHERE expires_at < NOW();
    $$
);


-- Verify after running:
-- SELECT jobid, jobname, schedule, command FROM cron.job WHERE jobname LIKE 'bean-%';