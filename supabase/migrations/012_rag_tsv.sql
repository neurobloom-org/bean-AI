-- Migration 012: Add full-text search vector to rag_techniques
ALTER TABLE public.rag_techniques
    ADD COLUMN IF NOT EXISTS name_description_tsv tsvector
        GENERATED ALWAYS AS (
            to_tsvector('english', coalesce(name, '') || ' ' || coalesce(description, ''))
        ) STORED;

CREATE INDEX IF NOT EXISTS idx_rag_techniques_tsv
    ON public.rag_techniques USING gin(name_description_tsv);