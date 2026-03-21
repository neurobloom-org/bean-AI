"""BEAN AI v5 — RAG service for CBT/DBT therapeutic techniques.

Retrieves relevant therapeutic techniques from a curated knowledge base
using vector similarity search. Used exclusively by the TherapyAgent.

The technique library is seeded in migration 005_rag_seed.sql.
Each technique has a pre-computed embedding stored in the rag_techniques table.

Privacy: technique retrieval is based on the user's *current message* only —
no personal history is passed to the retrieval query.
"""

import logging
from typing import Any

from services.embedding_service import get_embedding
from services.supabase_client import get_service_client

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback techniques (used when DB retrieval fails)
# ─────────────────────────────────────────────────────────────────────────────

_FALLBACK_TECHNIQUES = [
    {
        "name": "Reflective Listening",
        "description": "Mirror back what the user said to show understanding. "
        "Use phrases like 'It sounds like you're feeling...' or "
        "'What I'm hearing is...'",
        "example": "If they say 'nobody cares', reflect: 'It sounds like you're feeling "
        "really invisible right now. That's such a lonely feeling.'",
    },
    {
        "name": "Validation",
        "description": "Acknowledge that their feelings make sense given their situation. "
        "Avoid minimising or jumping to problem-solving.",
        "example": "'It makes complete sense you'd feel overwhelmed — that's a lot to "
        "carry at once.'",
    },
    {
        "name": "Open-ended questions",
        "description": "Ask questions that invite the user to explore their experience "
        "further. Avoid yes/no questions.",
        "example": "'What's that been like for you?' or 'Tell me more about what happened.'",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────────────────────────────────────


async def retrieve_cbt_techniques(
    situation_text: str,
    emotion: str,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Retrieve relevant CBT/DBT techniques for the current user situation.

    Args:
        situation_text: The user's current message (used as the query).
        emotion:        Detected emotion label (used to enrich the query).
        limit:          Number of techniques to return.

    Returns:
        List of technique dicts with keys: name, description, example.
    """
    if not situation_text.strip():
        return _FALLBACK_TECHNIQUES[:limit]

    # Build a semantically rich query
    query = (
        f"Therapeutic technique for someone who is {emotion}: {situation_text[:300]}"
    )

    try:
        embedding = await get_embedding(query)
    except Exception as exc:
        logger.warning("RAG: embedding failed, using fallback: %s", exc)
        return _FALLBACK_TECHNIQUES[:limit]

    try:
        client = await get_service_client()
        result = await client.rpc(
            "search_rag_techniques",
            {
                "p_query_embedding": embedding,
                "p_limit": limit,
                "p_min_similarity": 0.5,
            },
        ).execute()

        if result.data:
            return [
                {
                    "name": r.get("name", ""),
                    "description": r.get("description", ""),
                    "example": r.get("example", ""),
                }
                for r in result.data
            ]
        else:
            logger.debug("RAG: no techniques found for query, using fallback")
            return _FALLBACK_TECHNIQUES[:limit]

    except Exception as exc:
        logger.warning("RAG: retrieval failed, using fallback: %s", exc)
        return _FALLBACK_TECHNIQUES[:limit]


def format_techniques_for_prompt(techniques: list[dict[str, Any]]) -> str:
    """Format retrieved techniques into a readable prompt block.

    The TherapyAgent embeds this directly into its system prompt.
    The LLM is instructed NOT to name these techniques — just use them.
    """
    if not techniques:
        return "Use active listening, validation, and open-ended questions."

    parts = []
    for i, tech in enumerate(techniques, 1):
        name = tech.get("name", "Technique")
        desc = tech.get("description", "")
        example = tech.get("example", "")
        part = f"{i}. {name}: {desc}"
        if example:
            part += f"\n   Example: {example}"
        parts.append(part)

    return "\n\n".join(parts)
