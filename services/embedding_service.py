"""BEAN AI v5 — Embedding service (OpenAI text-embedding-3-small).

Used exclusively for episodic memory vector generation.

Privacy:
  - Text passed to this service is pre-truncated and redacted by PrivacyService
  - The source text is NOT stored anywhere after embedding
  - Only the resulting vector is persisted (in episodic_memories table)
"""

import asyncio
import logging

from openai import AsyncOpenAI

from shared.config import get_settings
from shared.exceptions import EmbeddingError

logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def get_embedding(text: str) -> list[float]:
    """Generate a vector embedding for the given text.

    Args:
        text: Input text (should already be truncated/redacted by caller).

    Returns:
        List of floats (1536-dimensional vector for text-embedding-3-small).

    Raises:
        EmbeddingError: If the OpenAI API call fails.
    """
    if not text or not text.strip():
        raise EmbeddingError("Cannot embed empty text")

    settings = get_settings()
    client = _get_client()

    try:
        response = await client.embeddings.create(
            model=settings.openai_embedding_model,
            input=text,
            dimensions=settings.openai_embedding_dimensions,
        )
        embedding = response.data[0].embedding
        logger.debug(
            "Embedding generated: dim=%d (source text NOT stored)",
            len(embedding),
        )
        return embedding

    except Exception as exc:
        logger.error("Embedding generation failed: %s", exc)
        raise EmbeddingError(f"Embedding failed: {exc}") from exc


async def search_similar_memories(
    user_id: str,
    query_text: str,
    top_k: int = 5,
    min_similarity: float = 0.7,
) -> list[dict]:
    """Search episodic memories by semantic similarity.

    Generates an embedding for query_text, then runs cosine similarity
    search against the user's episodic_memories in Supabase.

    Privacy: query_text is sent to OpenAI for embedding but NOT stored.

    Returns list of memory records (vector metadata only — no source text).
    """
    from services.supabase_client import get_service_client

    try:
        query_embedding = await get_embedding(query_text)
    except EmbeddingError as exc:
        logger.error("Memory search: embedding failed: %s", exc)
        return []

    try:
        client = await get_service_client()
        # Supabase supports pgvector similarity search via RPC
        result = await client.rpc(
            "search_episodic_memories",
            {
                "p_user_id": user_id,
                "p_query_embedding": query_embedding,
                "p_top_k": top_k,
                "p_min_similarity": min_similarity,
            },
        ).execute()
        return result.data or []

    except Exception as exc:
        logger.error("Memory search failed for user %s: %s", user_id, exc)
        return []
