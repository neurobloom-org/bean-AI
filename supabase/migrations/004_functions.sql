-- ============================================================================
-- BEAN AI v1 — Supabase Functions Migration 004
-- pgvector RPC functions + analytics RPC functions
-- ============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- search_episodic_memories
-- SECURITY INVOKER: correct — RLS enforces users only see their own memories
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION public.search_episodic_memories(
    p_user_id         UUID,
    p_query_embedding vector(1536),
    p_top_k           INTEGER DEFAULT 5,
    p_min_similarity  FLOAT   DEFAULT 0.7
)
RETURNS TABLE (
    id              UUID,
    session_id      UUID,
    emotion_label   TEXT,
    memory_type     TEXT,
    similarity      FLOAT,
    created_at      TIMESTAMPTZ
)
LANGUAGE sql
STABLE
SECURITY INVOKER
AS $$
    SELECT
        em.id,
        em.session_id,
        em.emotion_label,
        em.memory_type,
        1 - (em.embedding <=> p_query_embedding) AS similarity,
        em.created_at
    FROM public.episodic_memories em
    WHERE em.user_id = p_user_id
      AND (em.expires_at IS NULL OR em.expires_at > NOW())
      AND 1 - (em.embedding <=> p_query_embedding) >= p_min_similarity
    ORDER BY em.embedding <=> p_query_embedding
    LIMIT p_top_k;
$$;

COMMENT ON FUNCTION public.search_episodic_memories IS
    'Cosine similarity search over a user''s episodic memory vectors. Returns metadata only.';


-- ─────────────────────────────────────────────────────────────────────────────
-- search_rag_techniques
-- FIX: SECURITY DEFINER required — rag_techniques RLS blocks all authenticated
--      users (USING FALSE). SECURITY INVOKER would silently return 0 rows.
--      SET search_path = public is a security best practice with DEFINER.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION public.search_rag_techniques(
    p_query_embedding vector(1536),
    p_limit           INTEGER DEFAULT 3,
    p_min_similarity  FLOAT   DEFAULT 0.5
)
RETURNS TABLE (
    id          UUID,
    name        TEXT,
    description TEXT,
    example     TEXT,
    category    TEXT,
    similarity  FLOAT
)
LANGUAGE sql
STABLE
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT
        rt.id,
        rt.name,
        rt.description,
        rt.example,
        rt.category,
        1 - (rt.embedding <=> p_query_embedding) AS similarity
    FROM public.rag_techniques rt
    WHERE rt.embedding IS NOT NULL
      AND 1 - (rt.embedding <=> p_query_embedding) >= p_min_similarity
    ORDER BY rt.embedding <=> p_query_embedding
    LIMIT p_limit;
$$;

COMMENT ON FUNCTION public.search_rag_techniques IS
    'Cosine similarity search over CBT/DBT techniques. SECURITY DEFINER because rag_techniques blocks authenticated users via RLS.';


-- ─────────────────────────────────────────────────────────────────────────────
-- get_daily_emotion_summary
-- SECURITY INVOKER: correct — guardians can read patient emotion_events via
--                   the guardian RLS policy (can_view_graphs = TRUE)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION public.get_daily_emotion_summary(
    p_target_user_id    UUID,
    p_days              INTEGER,
    p_tz_offset_minutes INTEGER DEFAULT 0
)
RETURNS TABLE (
    day             text,
    emotion         text,
    count           bigint,
    avg_confidence  numeric
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

COMMENT ON FUNCTION public.get_daily_emotion_summary IS
    'Returns timezone-aware daily aggregated emotion counts for a user.';


-- ─────────────────────────────────────────────────────────────────────────────
-- get_weekly_session_activity
-- SECURITY INVOKER: correct — guardians can read patient sessions via
--                   the guardian RLS policy (can_view_graphs = TRUE)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION public.get_weekly_session_activity(
    p_target_user_id    UUID,
    p_weeks             INTEGER,
    p_tz_offset_minutes INTEGER DEFAULT 0
)
RETURNS TABLE (
    week                    text,
    session_count           bigint,
    total_duration_seconds  bigint,
    avg_turns_per_session   numeric,
    most_common_emotion     text
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

COMMENT ON FUNCTION public.get_weekly_session_activity IS
    'Returns timezone-aware weekly aggregated session activity for a user.';