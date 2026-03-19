-- ============================================================================
-- BEAN AI v5 — Supabase Functions Migration 004
-- pgvector RPC functions for semantic search
-- ============================================================================

-- ─────────────────────────────────────────────────────────────────────────────
-- search_episodic_memories
-- Called by embedding_service.search_similar_memories()
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
LANGUAGE sql STABLE
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
    'Cosine similarity search over a user''s episodic memory vectors. '
    'Returns metadata only — source text is never stored.';


-- ─────────────────────────────────────────────────────────────────────────────
-- search_rag_techniques
-- Called by services/rag_service.py retrieve_cbt_techniques()
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
LANGUAGE sql STABLE
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
    'Cosine similarity search over CBT/DBT techniques. '
    'Used by TherapyAgent to retrieve contextually relevant techniques.';


-- ─────────────────────────────────────────────────────────────────────────────
-- get_guardian_patient_overview
-- Called by guardian routes for the doctor dashboard
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION public.get_daily_emotion_summary(
    p_user_id  UUID,
    p_days     INTEGER DEFAULT 7
)
RETURNS TABLE (
    day             DATE,
    emotion         TEXT,
    count           BIGINT,
    avg_confidence  FLOAT
)
LANGUAGE sql STABLE
AS $$
    SELECT
        DATE(detected_at) AS day,
        emotion,
        COUNT(*) AS count,
        ROUND(AVG(confidence)::numeric, 2)::float AS avg_confidence
    FROM public.emotion_events
    WHERE user_id = p_user_id
      AND detected_at >= NOW() - (p_days || ' days')::interval
    GROUP BY DATE(detected_at), emotion
    ORDER BY day DESC, count DESC;
$$;
