-- Function to search Episodic Memories using Cosine Similarity
CREATE OR REPLACE FUNCTION search_episodic_memories (
  query_embedding VECTOR(1536),
  match_threshold FLOAT,
  match_count INT,
  p_user_id UUID
)
RETURNS TABLE (id UUID, content TEXT, similarity FLOAT)
LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  SELECT episodic_memory.id, episodic_memory.content,
         1 - (episodic_memory.embedding <=> query_embedding) AS similarity
  FROM episodic_memory
  WHERE episodic_memory.user_id = p_user_id
    AND 1 - (episodic_memory.embedding <=> query_embedding) > match_threshold
  ORDER BY episodic_memory.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;

-- Function to search Semantic Facts (Preferences/Facts)
CREATE OR REPLACE FUNCTION search_semantic_memories (
  query_embedding VECTOR(1536),
  match_threshold FLOAT,
  match_count INT,
  p_user_id UUID
)
RETURNS TABLE (id UUID, fact_text TEXT, similarity FLOAT)
LANGUAGE plpgsql AS $$
BEGIN
  RETURN QUERY
  SELECT semantic_memory.id, semantic_memory.fact_text,
         1 - (semantic_memory.embedding <=> query_embedding) AS similarity
  FROM semantic_memory
  WHERE semantic_memory.user_id = p_user_id
    AND 1 - (semantic_memory.embedding <=> query_embedding) > match_threshold
  ORDER BY semantic_memory.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;