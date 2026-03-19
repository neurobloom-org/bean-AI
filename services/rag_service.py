"""
services/rag_service.py
========================
RAG service for therapeutic response generation.
Needs embedding_service + supabase_client — must be last.
"""

from shared.config import config
from shared.enums import Framework
from services.embedding_service import generate_embedding
from services.supabase_client import supabase


async def retrieve_therapy_chunks(
    query: str,
    top_k: int = 3,
) -> list[dict]:
    """Retrieve relevant therapy chunks for a query."""
    query_embedding = await generate_embedding(query)
    result = supabase.rpc(
        "match_rag_techniques",
        {
            "query_embedding": query_embedding,
            "match_count": top_k,
        },
    ).execute()
    return result.data


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string."""
    context_parts = []
    for chunk in chunks:
        framework = chunk.get("framework", "general").upper()
        text = chunk.get("content", "")
        context_parts.append(f"[{framework}]\n{text}")
    return "\n\n---\n\n".join(context_parts)


def get_dominant_framework(chunks: list[dict]) -> Framework:
    """Get the most common framework from retrieved chunks."""
    if not chunks:
        return Framework.CBT
    frameworks = [c.get("framework", "cbt") for c in chunks]
    most_common = max(set(frameworks), key=frameworks.count)
    try:
        return Framework(most_common)
    except ValueError:
        return Framework.CBT


async def get_rag_context(query: str, top_k: int = 3) -> dict:
    """Get RAG context for a therapeutic query."""
    chunks = await retrieve_therapy_chunks(query, top_k)
    context = format_context(chunks)
    framework = get_dominant_framework(chunks)
    return {
        "context": context,
        "framework": framework,
        "chunks": chunks,
    }
