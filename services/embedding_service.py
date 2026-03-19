"""
services/embedding_service.py
==============================
Embedding service for generating text vectors.
Needs supabase_client — build after it.
"""

import google.generativeai as genai

from shared.config import config
from services.supabase_client import supabase

genai.configure(api_key=config.GOOGLE_API_KEY)


async def generate_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text."""
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
    )
    return result["embedding"]


async def store_embedding(
    table: str,
    record_id: str,
    embedding: list[float],
) -> None:
    """Store an embedding vector in Supabase."""
    supabase.table(table).update({"embedding": embedding}).eq("id", record_id).execute()


async def search_similar(
    table: str,
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict]:
    """Search for similar records using vector similarity."""
    result = supabase.rpc(
        "match_embeddings",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "table_name": table,
        },
    ).execute()
    return result.data
