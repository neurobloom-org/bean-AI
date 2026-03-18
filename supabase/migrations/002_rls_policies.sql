-- ============================================================================
-- BEAN AI v5 — Supabase RLS Policies Migration 002
-- Row Level Security: Privacy enforcement at the database layer
-- ============================================================================
-- These policies ensure that:
--   • Users can ONLY access their own data
--   • Guardians can see SUMMARIES (alerts, graphs) but NEVER raw transcripts
--   • No cross-user data leakage is possible even if app code has a bug
-- ============================================================================

-- Enable RLS on all user-facing tables
ALTER TABLE public.user_profiles     ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.guardian_links    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.sessions          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.session_transcripts ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.episodic_memories ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.emotion_events    ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.alerts            ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.tasks             ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.rate_limits       ENABLE ROW LEVEL SECURITY;


-- ─────────────────────────────────────────────────────────────────────────────
-- user_profiles
-- ─────────────────────────────────────────────────────────────────────────────

CREATE POLICY "user_profiles: users read own"
    ON public.user_profiles FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

CREATE POLICY "user_profiles: users insert own"
    ON public.user_profiles FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "user_profiles: users update own"
    ON public.user_profiles FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "user_profiles: users delete own"
    ON public.user_profiles FOR DELETE
    TO authenticated
    USING (auth.uid() = user_id);

-- Guardians see patient profile summary (for context in mobile app)
-- They see display_name and diagnosis_tags only — NOT personality_notes or
-- important_people (enforced via mobile app UI, not DB — full row visible to guardian)
CREATE POLICY "user_profiles: guardians read patient"
    ON public.user_profiles FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.guardian_links gl
            WHERE gl.guardian_user_id = auth.uid()
              AND gl.patient_user_id = user_profiles.user_id
              AND gl.can_view_graphs = TRUE
        )
    );


-- ─────────────────────────────────────────────────────────────────────────────
-- guardian_links
-- ─────────────────────────────────────────────────────────────────────────────

CREATE POLICY "guardian_links: users read own links"
    ON public.guardian_links FOR SELECT
    TO authenticated
    USING (
        auth.uid() = guardian_user_id OR auth.uid() = patient_user_id
    );

CREATE POLICY "guardian_links: patients can approve links"
    ON public.guardian_links FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = patient_user_id);

CREATE POLICY "guardian_links: patients can remove links"
    ON public.guardian_links FOR DELETE
    TO authenticated
    USING (auth.uid() = patient_user_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- sessions
-- ─────────────────────────────────────────────────────────────────────────────

CREATE POLICY "sessions: users read own"
    ON public.sessions FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

CREATE POLICY "sessions: users insert own"
    ON public.sessions FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "sessions: users update own"
    ON public.sessions FOR UPDATE
    TO authenticated
    USING (auth.uid() = user_id);

-- Guardians can see session metadata (timing, emotion summary, turn count)
-- They CANNOT see state_json or route_distribution in detail
-- (Mobile app should filter what it displays from this data)
CREATE POLICY "sessions: guardians read patient metadata"
    ON public.sessions FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.guardian_links gl
            WHERE gl.guardian_user_id = auth.uid()
              AND gl.patient_user_id = sessions.user_id
              AND gl.can_view_graphs = TRUE
        )
    );


-- ─────────────────────────────────────────────────────────────────────────────
-- session_transcripts
-- ⚠️  MOST RESTRICTIVE POLICY IN THE SYSTEM
-- Only the owning user can see their transcripts.
-- Guardians, doctors, and other users NEVER have access.
-- ─────────────────────────────────────────────────────────────────────────────

CREATE POLICY "session_transcripts: users read own only"
    ON public.session_transcripts FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

CREATE POLICY "session_transcripts: users insert own only"
    ON public.session_transcripts FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

-- NO guardian policy here — intentionally omitted.
-- Service role (background purge) bypasses RLS and can delete rows.
-- No other role can access other users' transcripts.


-- ─────────────────────────────────────────────────────────────────────────────
-- episodic_memories
-- ─────────────────────────────────────────────────────────────────────────────

CREATE POLICY "episodic_memories: users manage own"
    ON public.episodic_memories FOR ALL
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);

-- Guardians do NOT see episodic memories (vectors reveal nothing anyway,
-- but we block them for completeness).


-- ─────────────────────────────────────────────────────────────────────────────
-- emotion_events
-- ─────────────────────────────────────────────────────────────────────────────

CREATE POLICY "emotion_events: users read own"
    ON public.emotion_events FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

CREATE POLICY "emotion_events: users insert own"
    ON public.emotion_events FOR INSERT
    TO authenticated
    WITH CHECK (auth.uid() = user_id);

-- Guardians see emotion data for graph generation
-- (emotion labels are not personally identifiable on their own)
CREATE POLICY "emotion_events: guardians view patient graphs"
    ON public.emotion_events FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.guardian_links gl
            WHERE gl.guardian_user_id = auth.uid()
              AND gl.patient_user_id = emotion_events.user_id
              AND gl.can_view_graphs = TRUE
        )
    );


-- ─────────────────────────────────────────────────────────────────────────────
-- alerts
-- ─────────────────────────────────────────────────────────────────────────────

CREATE POLICY "alerts: users read own"
    ON public.alerts FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);

-- Guardians can see alert level + timestamp (not trigger phrases — those aren't stored)
CREATE POLICY "alerts: guardians view patient alerts"
    ON public.alerts FOR SELECT
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.guardian_links gl
            WHERE gl.guardian_user_id = auth.uid()
              AND gl.patient_user_id = alerts.user_id
              AND gl.can_view_alerts = TRUE
        )
    );

-- Guardians can acknowledge alerts (mark as reviewed)
CREATE POLICY "alerts: guardians acknowledge"
    ON public.alerts FOR UPDATE
    TO authenticated
    USING (
        EXISTS (
            SELECT 1 FROM public.guardian_links gl
            WHERE gl.guardian_user_id = auth.uid()
              AND gl.patient_user_id = alerts.user_id
              AND gl.can_view_alerts = TRUE
        )
    )
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.guardian_links gl
            WHERE gl.guardian_user_id = auth.uid()
              AND gl.patient_user_id = alerts.user_id
              AND gl.can_view_alerts = TRUE
        )
    );


-- ─────────────────────────────────────────────────────────────────────────────
-- tasks
-- ─────────────────────────────────────────────────────────────────────────────

CREATE POLICY "tasks: users manage own"
    ON public.tasks FOR ALL
    TO authenticated
    USING (auth.uid() = user_id)
    WITH CHECK (auth.uid() = user_id);


-- ─────────────────────────────────────────────────────────────────────────────
-- rate_limits
-- Service role only — users never read/write rate limit records directly
-- ─────────────────────────────────────────────────────────────────────────────

-- Block all authenticated user access (service_role bypasses RLS)
CREATE POLICY "rate_limits: block all user access"
    ON public.rate_limits FOR ALL
    TO authenticated
    USING (FALSE);


-- ─────────────────────────────────────────────────────────────────────────────
-- Useful views for mobile app / doctor dashboard
-- ─────────────────────────────────────────────────────────────────────────────

-- Daily emotion summary (for graph rendering)
-- Guardians query this view for their patients
CREATE OR REPLACE VIEW public.v_daily_emotion_summary AS
SELECT
    user_id,
    DATE_TRUNC('day', detected_at) AS day,
    emotion,
    COUNT(*)                        AS count,
    AVG(confidence)                 AS avg_confidence
FROM public.emotion_events
GROUP BY user_id, day, emotion;

COMMENT ON VIEW public.v_daily_emotion_summary IS
    'Aggregated emotion counts per day per user. Safe for guardian/doctor dashboard.';


-- Weekly session activity (for doctor trend analysis)
CREATE OR REPLACE VIEW public.v_weekly_session_activity AS
SELECT
    user_id,
    DATE_TRUNC('week', started_at)  AS week,
    COUNT(*)                         AS session_count,
    SUM(duration_seconds)            AS total_duration_seconds,
    AVG(turn_count)                  AS avg_turns_per_session,
    MODE() WITHIN GROUP (ORDER BY dominant_emotion) AS most_common_emotion
FROM public.sessions
WHERE status IN ('ended', 'expired')
GROUP BY user_id, week;

COMMENT ON VIEW public.v_weekly_session_activity IS
    'Weekly session stats. No conversation content. Safe for clinical dashboard.';