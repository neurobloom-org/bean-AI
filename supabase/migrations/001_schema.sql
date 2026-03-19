-- ============================================================================
-- BEAN AI v5 — Supabase Schema Migration 001
-- Privacy-First Mental Health Companion
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- ─────────────────────────────────────────────────────────────────────────────
-- user_profiles
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.user_profiles (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    display_name    TEXT,
    preferred_name  TEXT,

    interests           TEXT[]      NOT NULL DEFAULT '{}',
    important_people    TEXT[]      NOT NULL DEFAULT '{}',
    personality_notes   TEXT[]      NOT NULL DEFAULT '{}',

    -- Set by doctor/guardian onboarding — NOT extracted from conversation
    diagnosis_tags  TEXT[]      NOT NULL DEFAULT '{}',

    -- Safety flag: if true, F4_VULNERABILITY is always pre-set
    is_minor        BOOLEAN     NOT NULL DEFAULT FALSE,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_updated    TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT user_profiles_user_id_unique UNIQUE (user_id)
);

COMMENT ON COLUMN public.user_profiles.diagnosis_tags IS
    'Set by doctor/guardian only. Never inferred from conversation by BEAN.';
COMMENT ON COLUMN public.user_profiles.is_minor IS
    'If true, F4_VULNERABILITY is pre-set and alert threshold is lower.';


-- ─────────────────────────────────────────────────────────────────────────────
-- guardian_links
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.guardian_links (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    guardian_user_id    UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    patient_user_id     UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    relationship        TEXT        NOT NULL DEFAULT 'guardian',

    can_view_alerts     BOOLEAN     NOT NULL DEFAULT TRUE,
    can_view_graphs     BOOLEAN     NOT NULL DEFAULT TRUE,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT guardian_links_unique UNIQUE (guardian_user_id, patient_user_id),
    CONSTRAINT guardian_cannot_link_self CHECK (guardian_user_id != patient_user_id)
);


-- ─────────────────────────────────────────────────────────────────────────────
-- sessions
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.sessions (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    started_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at            TIMESTAMPTZ,
    duration_seconds    INTEGER,

    dominant_emotion    TEXT,
    turn_count          INTEGER     NOT NULL DEFAULT 0,
    route_distribution  JSONB       NOT NULL DEFAULT '{}',

    state_json          JSONB       NOT NULL DEFAULT '{}',

    status              TEXT        NOT NULL DEFAULT 'active',

    CONSTRAINT sessions_status_check CHECK (
        status IN ('active', 'ended', 'expired')
    )
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id    ON public.sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started_at ON public.sessions(started_at);
CREATE INDEX IF NOT EXISTS idx_sessions_status     ON public.sessions(status);


-- ─────────────────────────────────────────────────────────────────────────────
-- session_transcripts  (24h TTL)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.session_transcripts (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID        NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,
    user_id     UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    speaker     TEXT        NOT NULL,
    text        TEXT        NOT NULL,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL,

    CONSTRAINT transcript_expiry_valid  CHECK (expires_at > created_at),
    CONSTRAINT transcript_speaker_valid CHECK (speaker IN ('user', 'assistant'))
);

CREATE INDEX IF NOT EXISTS idx_session_transcripts_expires ON public.session_transcripts(expires_at);
CREATE INDEX IF NOT EXISTS idx_session_transcripts_session ON public.session_transcripts(session_id, created_at);


-- ─────────────────────────────────────────────────────────────────────────────
-- episodic_memories  (vectors only)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.episodic_memories (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id      UUID        REFERENCES public.sessions(id) ON DELETE SET NULL,

    embedding       vector(1536) NOT NULL,

    emotion_label   TEXT,
    memory_type     TEXT        NOT NULL DEFAULT 'episodic',
    relevance_score FLOAT       NOT NULL DEFAULT 1.0,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at      TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_episodic_user ON public.episodic_memories(user_id);
CREATE INDEX IF NOT EXISTS idx_episodic_embedding
    ON public.episodic_memories
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);


-- ─────────────────────────────────────────────────────────────────────────────
-- emotion_events
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.emotion_events (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id  UUID        NOT NULL REFERENCES public.sessions(id) ON DELETE CASCADE,

    emotion     TEXT        NOT NULL,
    confidence  FLOAT,

    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '90 days')
);

CREATE INDEX IF NOT EXISTS idx_emotion_user_time ON public.emotion_events(user_id, detected_at);
CREATE INDEX IF NOT EXISTS idx_emotion_session   ON public.emotion_events(session_id);
CREATE INDEX IF NOT EXISTS idx_emotion_expires   ON public.emotion_events(expires_at);


-- ─────────────────────────────────────────────────────────────────────────────
-- alerts
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.alerts (
    id                      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                 UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id              UUID        REFERENCES public.sessions(id) ON DELETE SET NULL,

    alert_level             TEXT        NOT NULL,
    alert_factors           TEXT[]      NOT NULL DEFAULT '{}',

    notified_guardian       BOOLEAN     NOT NULL DEFAULT FALSE,
    guardian_notified_at    TIMESTAMPTZ,
    sms_sent                BOOLEAN     NOT NULL DEFAULT FALSE,
    acknowledged            BOOLEAN     NOT NULL DEFAULT FALSE,
    acknowledged_at         TIMESTAMPTZ,

    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT alerts_level_check CHECK (
        alert_level IN ('low', 'medium', 'high', 'crisis')
    )
);

CREATE INDEX IF NOT EXISTS idx_alerts_user          ON public.alerts(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_unacknowledged ON public.alerts(user_id) WHERE acknowledged = FALSE;


-- ─────────────────────────────────────────────────────────────────────────────
-- tasks  (FIX: added reminder_at and calendar_event_id columns)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.tasks (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id             UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,

    title               TEXT        NOT NULL,
    description         TEXT,
    due_at              TIMESTAMPTZ,
    reminder_at         TIMESTAMPTZ,          -- when to alert user (due_at - 10min)
    calendar_event_id   TEXT,                 -- Google Calendar event ID (nullable)

    status              TEXT        NOT NULL DEFAULT 'pending',

    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT tasks_status_check CHECK (
        status IN ('pending', 'reminded', 'completed', 'snoozed', 'cancelled')
    )
);

CREATE INDEX IF NOT EXISTS idx_tasks_user_due ON public.tasks(user_id, due_at)
    WHERE status IN ('pending', 'snoozed');
CREATE INDEX IF NOT EXISTS idx_tasks_reminder ON public.tasks(reminder_at)
    WHERE status IN ('pending', 'snoozed') AND reminder_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_tasks_calendar_event ON public.tasks(calendar_event_id)
    WHERE calendar_event_id IS NOT NULL;


-- ─────────────────────────────────────────────────────────────────────────────
-- rate_limits
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.rate_limits (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    key             TEXT        NOT NULL UNIQUE,
    window_start    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_count   INTEGER     NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_rate_limits_key    ON public.rate_limits(key);
CREATE INDEX IF NOT EXISTS idx_rate_limits_window ON public.rate_limits(window_start);


-- ─────────────────────────────────────────────────────────────────────────────
-- oauth_tokens  (Google Calendar OAuth — NEW)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.oauth_tokens (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID        NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    provider        TEXT        NOT NULL DEFAULT 'google_calendar',
    access_token    TEXT        NOT NULL,
    refresh_token   TEXT,
    expires_at      TIMESTAMPTZ,
    scope           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT oauth_tokens_user_provider UNIQUE (user_id, provider)
);

COMMENT ON TABLE public.oauth_tokens IS
    'OAuth tokens for third-party integrations. Access tokens encrypted at rest by Supabase.';


-- ─────────────────────────────────────────────────────────────────────────────
-- tts_cache  (pre-cached filler phrase audio — NEW)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.tts_cache (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    cache_key   TEXT        NOT NULL UNIQUE,
    phrase      TEXT        NOT NULL,
    voice_id    TEXT        NOT NULL,
    audio_b64   TEXT        NOT NULL,    -- base64-encoded PCM audio
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '7 days')
);

CREATE INDEX IF NOT EXISTS idx_tts_cache_key     ON public.tts_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_tts_cache_expires ON public.tts_cache(expires_at);


-- ─────────────────────────────────────────────────────────────────────────────
-- rag_techniques  (CBT/DBT knowledge base for therapy RAG — NEW)
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.rag_techniques (
    id          UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT        NOT NULL,
    description TEXT        NOT NULL,
    example     TEXT,
    category    TEXT        NOT NULL DEFAULT 'cbt',   -- 'cbt' | 'dbt' | 'general'
    embedding   vector(1536),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rag_techniques_embedding
    ON public.rag_techniques
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 10);


-- ─────────────────────────────────────────────────────────────────────────────
-- health_check
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.health_check (
    id          INTEGER     PRIMARY KEY DEFAULT 1,
    checked_at  TIMESTAMPTZ DEFAULT NOW()
);
INSERT INTO public.health_check (id) VALUES (1) ON CONFLICT DO NOTHING;


-- ─────────────────────────────────────────────────────────────────────────────
-- Triggers
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION public.update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER tasks_updated_at
    BEFORE UPDATE ON public.tasks
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER profiles_updated_at
    BEFORE UPDATE ON public.user_profiles
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();

CREATE TRIGGER oauth_tokens_updated_at
    BEFORE UPDATE ON public.oauth_tokens
    FOR EACH ROW EXECUTE FUNCTION public.update_updated_at();
