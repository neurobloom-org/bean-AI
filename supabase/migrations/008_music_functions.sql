-- ============================================================================
-- BEAN AI — Migration 008: Helper functions for music
-- ============================================================================

-- Atomic play count increment (avoids read-modify-write race)
CREATE OR REPLACE FUNCTION public.increment_song_play_count(p_song_id UUID)
RETURNS VOID AS $$
BEGIN
    UPDATE public.user_songs
    SET play_count = play_count + 1
    WHERE id = p_song_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

COMMENT ON FUNCTION public.increment_song_play_count IS
    'Atomically increments the play count for a song. Called by the backend after a song finishes streaming.';