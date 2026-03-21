-- ============================================================================
-- BEAN AI v5 — Auth / OAuth / Emotion Fixes Migration 006
-- Adds profile auto-create trigger and reasserts critical auth-related DB logic
-- ============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- Auto-create user_profiles row when auth.users row is created
-- display_name is taken from raw_user_meta_data
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION public.handle_new_user_profile()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    INSERT INTO public.user_profiles (
        user_id,
        display_name
    )
    VALUES (
        NEW.id,
        NEW.raw_user_meta_data ->> 'display_name'
    )
    ON CONFLICT (user_id) DO NOTHING;

    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created_profile ON auth.users;

CREATE TRIGGER on_auth_user_created_profile
AFTER INSERT ON auth.users
FOR EACH ROW
EXECUTE FUNCTION public.handle_new_user_profile();


-- ─────────────────────────────────────────────────────────────────────────────
-- Backfill missing profiles for any existing auth users
-- ─────────────────────────────────────────────────────────────────────────────
INSERT INTO public.user_profiles (user_id, display_name)
SELECT
    au.id,
    au.raw_user_meta_data ->> 'display_name'
FROM auth.users au
LEFT JOIN public.user_profiles up
    ON up.user_id = au.id
WHERE up.user_id IS NULL;


-- ─────────────────────────────────────────────────────────────────────────────
-- Reassert oauth_tokens RLS in case this migration is run independently later
-- ─────────────────────────────────────────────────────────────────────────────
ALTER TABLE public.oauth_tokens ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "oauth_tokens: users read own" ON public.oauth_tokens;
CREATE POLICY "oauth_tokens: users read own"
    ON public.oauth_tokens FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "oauth_tokens: users delete own" ON public.oauth_tokens;
CREATE POLICY "oauth_tokens: users delete own"
    ON public.oauth_tokens FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "oauth_tokens: users update own" ON public.oauth_tokens;
CREATE POLICY "oauth_tokens: users update own"
    ON public.oauth_tokens FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- Recreate analytics functions with final signatures
-- safe to rerun because CREATE OR REPLACE FUNCTION is idempotent
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION public.get_daily_emotion_summary(
    p_target_user_id UUID,
    p_days INTEGER,
    p_tz_offset_minutes INTEGER DEFAULT 0
)
RETURNS TABLE (
    day text,
    emotion text,
    count bigint,
    avg_confidence numeric
)
LANGUAGE sql
STABLE
SECURITY INVOKER
AS $$
    WITH shifted_events AS (
        SELECT
            (
                detected_at + make_interval(mins => p_tz_offset_minutes)
            )::date AS local_day,
            emotion,
            COALESCE(confidence, 0)::numeric AS confidence
        FROM public.emotion_events
        WHERE user_id = p_target_user_id
          AND detected_at > NOW() - make_interval(days => p_days)
    )
    SELECT
        TO_CHAR(local_day, 'YYYY-MM-DD') AS day,
        emotion,
        COUNT(*) AS count,
        ROUND(AVG(confidence), 2) AS avg_confidence
    FROM shifted_events
    GROUP BY local_day, emotion
    ORDER BY local_day ASC, emotion ASC;
$$;

CREATE OR REPLACE FUNCTION public.get_weekly_session_activity(
    p_target_user_id UUID,
    p_weeks INTEGER,
    p_tz_offset_minutes INTEGER DEFAULT 0
)
RETURNS TABLE (
    week text,
    session_count bigint,
    total_duration_seconds bigint,
    avg_turns_per_session numeric,
    most_common_emotion text
)
LANGUAGE sql
STABLE
SECURITY INVOKER
AS $$
    WITH shifted_sessions AS (
        SELECT
            (
                started_at + make_interval(mins => p_tz_offset_minutes)
            ) AS local_started_at,
            COALESCE(duration_seconds, 0)::bigint AS duration_seconds,
            COALESCE(turn_count, 0)::bigint AS turn_count,
            dominant_emotion
        FROM public.sessions
        WHERE user_id = p_target_user_id
          AND started_at > NOW() - make_interval(days => p_weeks * 7)
          AND status IN ('ended', 'expired')
    ),
    bucketed AS (
        SELECT
            DATE_TRUNC('week', local_started_at)::date AS week_start,
            duration_seconds,
            turn_count,
            dominant_emotion
        FROM shifted_sessions
    ),
    emotion_rank AS (
        SELECT
            week_start,
            dominant_emotion,
            COUNT(*) AS emotion_count,
            ROW_NUMBER() OVER (
                PARTITION BY week_start
                ORDER BY COUNT(*) DESC, dominant_emotion ASC
            ) AS rn
        FROM bucketed
        WHERE dominant_emotion IS NOT NULL
        GROUP BY week_start, dominant_emotion
    ),
    weekly_stats AS (
        SELECT
            week_start,
            COUNT(*) AS session_count,
            SUM(duration_seconds) AS total_duration_seconds,
            ROUND(AVG(turn_count), 1) AS avg_turns_per_session
        FROM bucketed
        GROUP BY week_start
    )
    SELECT
        TO_CHAR(ws.week_start, 'YYYY-MM-DD') AS week,
        ws.session_count,
        ws.total_duration_seconds,
        ws.avg_turns_per_session,
        er.dominant_emotion AS most_common_emotion
    FROM weekly_stats ws
    LEFT JOIN emotion_rank er
        ON ws.week_start = er.week_start
       AND er.rn = 1
    ORDER BY ws.week_start ASC;
$$;