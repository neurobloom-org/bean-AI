-- ============================================================================
-- BEAN AI v1 — Migration 009: Atomic Rate Limit RPC
-- ============================================================================
-- Replaces the non-atomic Python read-then-update pattern in
-- api/middleware/rate_limiter.py with a single atomic Postgres statement.
--
-- The INSERT … ON CONFLICT DO UPDATE executes as a single statement under
-- Postgres's MVCC — there is no window between the read and the write where
-- a concurrent session can interleave, so the count is always correct even
-- under high concurrency across multiple replicas.
-- ============================================================================

CREATE OR REPLACE FUNCTION public.check_and_increment_rate_limit(
    p_key            TEXT,
    p_window_seconds INTEGER
)
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    v_count INTEGER;
BEGIN
    INSERT INTO public.rate_limits (key, request_count, window_start)
    VALUES (p_key, 1, NOW())
    ON CONFLICT (key) DO UPDATE
        SET
            -- If the stored window is still active, increment the counter.
            -- If it has expired, start a fresh window at count = 1.
            request_count = CASE
                WHEN rate_limits.window_start
                         > (NOW() - (p_window_seconds || ' seconds')::INTERVAL)
                THEN rate_limits.request_count + 1
                ELSE 1
            END,
            window_start = CASE
                WHEN rate_limits.window_start
                         > (NOW() - (p_window_seconds || ' seconds')::INTERVAL)
                THEN rate_limits.window_start   -- preserve original window start
                ELSE NOW()                       -- open a new window
            END
    RETURNING request_count INTO v_count;

    RETURN v_count;
END;
$$;

COMMENT ON FUNCTION public.check_and_increment_rate_limit(TEXT, INTEGER) IS
    'Atomically upsert a rate-limit counter. Returns the count after increment. '
    'Caller compares the returned count against the configured limit.';

-- Grant execute to the service role used by BEAN AI.
-- The anon role must NOT have access — rate-limit enforcement is server-side only.
REVOKE EXECUTE ON FUNCTION public.check_and_increment_rate_limit(TEXT, INTEGER)
    FROM PUBLIC;

GRANT EXECUTE ON FUNCTION public.check_and_increment_rate_limit(TEXT, INTEGER)
    TO service_role;