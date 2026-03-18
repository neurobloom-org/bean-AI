-- Enable the cron extension
CREATE EXTENSION IF NOT EXISTS pg_cron;

-- 1. Purge session transcripts every hour (privacy requirement)
SELECT cron.schedule('purge-transcripts', '0 * * * *', $$
  DELETE FROM session_transcripts WHERE created_at < NOW() - INTERVAL '1 hour';
$$);

-- 2. Clear old emotion logs every day
SELECT cron.schedule('daily-emotion-cleanup', '0 0 * * *', $$
  DELETE FROM emotion_logs WHERE created_at < NOW() - INTERVAL '30 days';
$$);

-- 3. Cleanup rate limits every hour
SELECT cron.schedule('reset-rate-limits', '0 * * * *', $$
  DELETE FROM rate_limits WHERE last_request < NOW() - INTERVAL '1 hour';
$$);