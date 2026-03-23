-- ============================================================================
-- BEAN AI v1 — Migration 007: User Songs & Music Storage
-- ============================================================================
-- Songs are stored in Supabase Storage bucket "bean-music".
-- Metadata lives in this table.
--
-- Storage layout:
--   bean-music/
--     defaults/{mood}/{filename}.mp3   ← pre-loaded by admin, visible to all
--     user_uploads/{user_id}/{id}.mp3  ← user-uploaded, private
--
-- After running this migration:
--   1. Create the storage bucket in Supabase dashboard:
--      Storage → New bucket → Name: "bean-music" → NOT public → Save
--   2. Upload default calm songs to: defaults/calm/
--      Upload default happy songs to: defaults/happy/
--      ... etc for each mood folder
--   3. Run: INSERT INTO user_songs (...) for each default song
--      OR use the seed script: scripts/seed_default_songs.py
-- ============================================================================

-- ── user_songs table ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.user_songs (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID        REFERENCES auth.users(id) ON DELETE CASCADE,
    -- user_id is NULL for default songs (admin-uploaded, shared with all users)

    title           TEXT        NOT NULL,
    mood            TEXT        NOT NULL,   -- calm | happy | sad | lofi | nature | classical

    -- Path inside the "bean-music" Supabase Storage bucket
    storage_path    TEXT        NOT NULL UNIQUE,

    file_size_bytes INTEGER,
    duration_seconds FLOAT,
    mime_type       TEXT        NOT NULL DEFAULT 'audio/mpeg',

    -- TRUE = shipped with BEAN, visible to all users
    -- FALSE = uploaded by this user, visible only to them
    is_default      BOOLEAN     NOT NULL DEFAULT FALSE,

    play_count      INTEGER     NOT NULL DEFAULT 0,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT user_songs_mood_check CHECK (
        mood IN ('calm', 'happy', 'sad', 'lofi', 'nature', 'classical')
    ),
    CONSTRAINT user_songs_mime_check CHECK (
        mime_type IN ('audio/mpeg', 'audio/ogg', 'audio/wav', 'audio/mp4', 'audio/aac')
    ),
    -- Non-default songs must have a user_id
    CONSTRAINT user_songs_owner_check CHECK (
        is_default = TRUE OR user_id IS NOT NULL
    )
);

COMMENT ON TABLE  public.user_songs IS
    'Metadata for songs stored in Supabase Storage (bean-music bucket).';
COMMENT ON COLUMN public.user_songs.storage_path IS
    'Path inside the bean-music bucket, e.g. defaults/calm/01.mp3';
COMMENT ON COLUMN public.user_songs.is_default IS
    'Admin-seeded songs visible to all users. user_id is NULL for these.';

CREATE INDEX IF NOT EXISTS idx_user_songs_user_mood
    ON public.user_songs(user_id, mood)
    WHERE user_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_user_songs_defaults
    ON public.user_songs(mood)
    WHERE is_default = TRUE;


-- ── Row Level Security ────────────────────────────────────────────────────────

ALTER TABLE public.user_songs ENABLE ROW LEVEL SECURITY;

-- Authenticated users can read their own songs AND all default songs
DROP POLICY IF EXISTS "user_songs: read own and defaults" ON public.user_songs;
CREATE POLICY "user_songs: read own and defaults"
    ON public.user_songs FOR SELECT
    TO authenticated
    USING (
        is_default = TRUE
        OR auth.uid() = user_id
    );

-- Users can upload songs for themselves only
DROP POLICY IF EXISTS "user_songs: insert own" ON public.user_songs;
CREATE POLICY "user_songs: insert own"
    ON public.user_songs FOR INSERT
    TO authenticated
    WITH CHECK (
        is_default = FALSE
        AND auth.uid() = user_id
    );

-- Users can delete their own non-default songs
DROP POLICY IF EXISTS "user_songs: delete own" ON public.user_songs;
CREATE POLICY "user_songs: delete own"
    ON public.user_songs FOR DELETE
    TO authenticated
    USING (
        is_default = FALSE
        AND auth.uid() = user_id
    );

-- Service role can do anything (for admin seeding and background jobs)
DROP POLICY IF EXISTS "user_songs: service role full access" ON public.user_songs;
CREATE POLICY "user_songs: service role full access"
    ON public.user_songs FOR ALL
    TO service_role
    USING (TRUE)
    WITH CHECK (TRUE);


-- ── Storage bucket policies ───────────────────────────────────────────────────
-- Run these in Supabase SQL editor AFTER creating the bucket via the dashboard.
-- The bucket must be created first (INSERT into storage.buckets won't work from migrations).

-- Allow authenticated users to read defaults/* (shared songs)
-- Allow authenticated users to read/write their own user_uploads/{uid}/* paths
-- Allow service role to manage everything

-- NOTE: These storage policies are set in the Supabase dashboard under
-- Storage → bean-music → Policies, or via the SQL editor:

/*
-- Paste this into Supabase SQL editor after creating the "bean-music" bucket:

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'bean-music',
    'bean-music',
    false,                        -- private bucket (access via signed URLs)
    20971520,                     -- 20 MB per file max
    ARRAY['audio/mpeg', 'audio/ogg', 'audio/wav', 'audio/mp4', 'audio/aac']
) ON CONFLICT (id) DO NOTHING;

-- Service role: full access for streaming + admin uploads
CREATE POLICY "bean-music: service role full access"
    ON storage.objects FOR ALL TO service_role
    USING (bucket_id = 'bean-music')
    WITH CHECK (bucket_id = 'bean-music');

-- Authenticated users: read defaults (shared BEAN songs)
CREATE POLICY "bean-music: read defaults"
    ON storage.objects FOR SELECT TO authenticated
    USING (
        bucket_id = 'bean-music'
        AND (storage.foldername(name))[1] = 'defaults'
    );

-- Authenticated users: read/write/delete their own uploads
CREATE POLICY "bean-music: manage own uploads"
    ON storage.objects FOR ALL TO authenticated
    USING (
        bucket_id = 'bean-music'
        AND (storage.foldername(name))[1] = 'user_uploads'
        AND (storage.foldername(name))[2] = auth.uid()::text
    )
    WITH CHECK (
        bucket_id = 'bean-music'
        AND (storage.foldername(name))[1] = 'user_uploads'
        AND (storage.foldername(name))[2] = auth.uid()::text
    );
*/


-- ── Default songs placeholder ─────────────────────────────────────────────────
-- After uploading files to Supabase Storage, insert metadata rows here.
-- Example (run after uploading the file):
--
-- INSERT INTO public.user_songs (title, mood, storage_path, is_default, mime_type, duration_seconds)
-- VALUES
--   ('Gentle Morning', 'calm', 'defaults/calm/gentle_morning.mp3', TRUE, 'audio/mpeg', 187),
--   ('Ocean Breeze',   'calm', 'defaults/calm/ocean_breeze.mp3',   TRUE, 'audio/mpeg', 210),
--   ('Forest Rain',    'calm', 'defaults/calm/forest_rain.mp3',    TRUE, 'audio/mpeg', 240),
--   ('Soft Piano',     'calm', 'defaults/calm/soft_piano.mp3',     TRUE, 'audio/mpeg', 195),
--   ('Happy Day',      'happy','defaults/happy/happy_day.mp3',     TRUE, 'audio/mpeg', 162),
--   ('Morning Run',    'happy','defaults/happy/morning_run.mp3',   TRUE, 'audio/mpeg', 178),
--   ('Study Beats',    'lofi', 'defaults/lofi/study_beats.mp3',    TRUE, 'audio/mpeg', 220),
--   ('Night Coding',   'lofi', 'defaults/lofi/night_coding.mp3',   TRUE, 'audio/mpeg', 235)
-- ON CONFLICT DO NOTHING;