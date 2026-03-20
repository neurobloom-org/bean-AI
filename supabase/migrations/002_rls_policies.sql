-- ============================================================================
-- BEAN AI v1 — Supabase RLS Policies Migration 002
-- Row Level Security: Privacy enforcement at the database layer
-- ============================================================================

ALTER TABLE public.user_profiles         ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.guardian_links        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions              ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.session_transcripts   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.episodic_memories     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.emotion_events        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alerts                ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tasks                 ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rate_limits           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.oauth_tokens          ENABLE ROW LEVEL SECURITY;

-- Optional hardening for non-user-facing tables
ALTER TABLE public.tts_cache             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rag_techniques        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.health_check          ENABLE ROW LEVEL SECURITY;


-- ─────────────────────────────────────────────────────────────────────────────
-- user_profiles
-- ─────────────────────────────────────────────────────────────────────────────
DROP POLICY IF EXISTS "user_profiles: users read own" ON public.user_profiles;
CREATE POLICY "user_profiles: users read own"
    ON public.user_profiles FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "user_profiles: users insert own" ON public.user_profiles;
CREATE POLICY "user_profiles: users insert own"
    ON public.user_profiles FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "user_profiles: users update own" ON public.user_profiles;
CREATE POLICY "user_profiles: users update own"
    ON public.user_profiles FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "user_profiles: users delete own" ON public.user_profiles;
CREATE POLICY "user_profiles: users delete own"
    ON public.user_profiles FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "user_profiles: guardians read patient" ON public.user_profiles;
CREATE POLICY "user_profiles: guardians read patient"
    ON public.user_profiles FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1
            FROM public.guardian_links gl
            WHERE gl.guardian_user_id = auth.uid()
              AND gl.patient_user_id = user_profiles.user_id
              AND gl.can_view_graphs = TRUE
        )
    );


-- ─────────────────────────────────────────────────────────────────────────────
-- guardian_links
-- ─────────────────────────────────────────────────────────────────────────────
DROP POLICY IF EXISTS "guardian_links: users read own links" ON public.guardian_links;
CREATE POLICY "guardian_links: users read own links"
    ON public.guardian_links FOR SELECT
    TO authenticated
    USING (
        auth.uid() = guardian_user_id OR auth.uid() = patient_user_id
    );

DROP POLICY IF EXISTS "guardian_links: patients can approve links" ON public.guardian_links;
CREATE POLICY "guardian_links: patients can approve links"
    ON public.guardian_links FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = patient_user_id);

DROP POLICY IF EXISTS "guardian_links: patients can remove links" ON public.guardian_links;
CREATE POLICY "guardian_links: patients can remove links"
    ON public.guardian_links FOR DELETE
    TO authenticated
    USING (auth.uid() = patient_user_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- sessions
-- ─────────────────────────────────────────────────────────────────────────────
DROP POLICY IF EXISTS "sessions: users read own" ON public.sessions;
CREATE POLICY "sessions: users read own"
    ON public.sessions FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "sessions: users insert own" ON public.sessions;
CREATE POLICY "sessions: users insert own"
    ON public.sessions FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "sessions: users update own" ON public.sessions;
CREATE POLICY "sessions: users update own"
    ON public.sessions FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "sessions: guardians read patient metadata" ON public.sessions;
CREATE POLICY "sessions: guardians read patient metadata"
    ON public.sessions FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1
            FROM public.guardian_links gl
            WHERE gl.guardian_user_id = auth.uid()
              AND gl.patient_user_id = sessions.user_id
              AND gl.can_view_graphs = TRUE
        )
    );


-- ─────────────────────────────────────────────────────────────────────────────
-- session_transcripts
-- ─────────────────────────────────────────────────────────────────────────────
DROP POLICY IF EXISTS "session_transcripts: users read own only" ON public.session_transcripts;
CREATE POLICY "session_transcripts: users read own only"
    ON public.session_transcripts FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "session_transcripts: users insert own only" ON public.session_transcripts;
CREATE POLICY "session_transcripts: users insert own only"
    ON public.session_transcripts FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- episodic_memories
-- ─────────────────────────────────────────────────────────────────────────────
DROP POLICY IF EXISTS "episodic_memories: users manage own" ON public.episodic_memories;
CREATE POLICY "episodic_memories: users manage own"
    ON public.episodic_memories FOR ALL
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- emotion_events
-- ─────────────────────────────────────────────────────────────────────────────
DROP POLICY IF EXISTS "emotion_events: users read own" ON public.emotion_events;
CREATE POLICY "emotion_events: users read own"
    ON public.emotion_events FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "emotion_events: users insert own" ON public.emotion_events;
CREATE POLICY "emotion_events: users insert own"
    ON public.emotion_events FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "emotion_events: guardians view patient graphs" ON public.emotion_events;
CREATE POLICY "emotion_events: guardians view patient graphs"
    ON public.emotion_events FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1
            FROM public.guardian_links gl
            WHERE gl.guardian_user_id = auth.uid()
              AND gl.patient_user_id = emotion_events.user_id
              AND gl.can_view_graphs = TRUE
        )
    );


-- ─────────────────────────────────────────────────────────────────────────────
-- alerts
-- ─────────────────────────────────────────────────────────────────────────────
DROP POLICY IF EXISTS "alerts: users read own" ON public.alerts;
CREATE POLICY "alerts: users read own"
    ON public.alerts FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "alerts: users acknowledge own" ON public.alerts;
CREATE POLICY "alerts: users acknowledge own"
    ON public.alerts FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

DROP POLICY IF EXISTS "alerts: guardians view patient alerts" ON public.alerts;
CREATE POLICY "alerts: guardians view patient alerts"
    ON public.alerts FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1
            FROM public.guardian_links gl
            WHERE gl.guardian_user_id = auth.uid()
              AND gl.patient_user_id = alerts.user_id
              AND gl.can_view_alerts = TRUE
        )
    );

DROP POLICY IF EXISTS "alerts: guardians acknowledge" ON public.alerts;
CREATE POLICY "alerts: guardians acknowledge"
    ON public.alerts FOR UPDATE
    TO authenticated
    USING (
        EXISTS (
            SELECT 1
            FROM public.guardian_links gl
            WHERE gl.guardian_user_id = auth.uid()
              AND gl.patient_user_id = alerts.user_id
              AND gl.can_view_alerts = TRUE
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1
            FROM public.guardian_links gl
            WHERE gl.guardian_user_id = auth.uid()
              AND gl.patient_user_id = alerts.user_id
              AND gl.can_view_alerts = TRUE
        )
    );


-- ─────────────────────────────────────────────────────────────────────────────
-- tasks
-- ─────────────────────────────────────────────────────────────────────────────
DROP POLICY IF EXISTS "tasks: users manage own" ON public.tasks;
CREATE POLICY "tasks: users manage own"
    ON public.tasks FOR ALL
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- rate_limits
-- ─────────────────────────────────────────────────────────────────────────────
DROP POLICY IF EXISTS "rate_limits: block all user access" ON public.rate_limits;
CREATE POLICY "rate_limits: block all user access"
    ON public.rate_limits FOR ALL
    TO authenticated
    USING (FALSE)
    WITH CHECK (FALSE);


-- ─────────────────────────────────────────────────────────────────────────────
-- oauth_tokens
-- ─────────────────────────────────────────────────────────────────────────────
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
-- tts_cache / rag_techniques / health_check
-- block authenticated direct access — service role only
-- ─────────────────────────────────────────────────────────────────────────────
DROP POLICY IF EXISTS "tts_cache: block all user access" ON public.tts_cache;
CREATE POLICY "tts_cache: block all user access"
    ON public.tts_cache FOR ALL
    TO authenticated
    USING (FALSE)
    WITH CHECK (FALSE);

DROP POLICY IF EXISTS "rag_techniques: block all user access" ON public.rag_techniques;
CREATE POLICY "rag_techniques: block all user access"
    ON public.rag_techniques FOR ALL
    TO authenticated
    USING (FALSE)
    WITH CHECK (FALSE);

DROP POLICY IF EXISTS "health_check: block all user access" ON public.health_check;
CREATE POLICY "health_check: block all user access"
    ON public.health_check FOR ALL
    TO authenticated
    USING (FALSE)
    WITH CHECK (FALSE);


-- ─────────────────────────────────────────────────────────────────────────────
-- Useful views
-- FIX: security_invoker = on ensures RLS on underlying tables is enforced
--      when users query these views. Without this, views can bypass RLS
--      in Supabase's PostgreSQL environment.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW public.v_daily_emotion_summary
WITH (security_invoker = on) AS
SELECT
    user_id,
    DATE_TRUNC('day', detected_at) AS day,
    emotion,
    COUNT(*) AS count,
    AVG(confidence) AS avg_confidence
FROM public.emotion_events
GROUP BY user_id, DATE_TRUNC('day', detected_at), emotion;

COMMENT ON VIEW public.v_daily_emotion_summary IS
    'Aggregated emotion counts per day per user. Safe for guardian/doctor dashboard.';


CREATE OR REPLACE VIEW public.v_weekly_session_activity
WITH (security_invoker = on) AS
SELECT
    user_id,
    DATE_TRUNC('week', started_at) AS week,
    COUNT(*) AS session_count,
    SUM(duration_seconds) AS total_duration_seconds,
    AVG(turn_count) AS avg_turns_per_session,
    MODE() WITHIN GROUP (ORDER BY dominant_emotion) AS most_common_emotion
FROM public.sessions
WHERE status IN ('ended', 'expired')
GROUP BY user_id, DATE_TRUNC('week', started_at);

COMMENT ON VIEW public.v_weekly_session_activity IS
    'Weekly session stats. No conversation content. Safe for clinical dashboard.';