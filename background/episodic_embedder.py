"""BEAN AI — Episodic embedder background job.

Disabled.

The original implementation attempted to generate embeddings for
episodic_memories rows where embedding is NULL by reading memory_text
from the database. That column does not exist in the episodic_memories
table by design, because raw episodic text is not stored for privacy
reasons.

Until a correct retry architecture exists (for example, a queue table
or another non-privacy-breaking fallback), this background job is a
deliberate no-op and should not be scheduled from api/main.py.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def embed_missing_episodic_memories() -> int:
    """Disabled: episodic memories do not store raw text to embed."""
    logger.warning(
        "[EpisodicEmbedder] Disabled: episodic_memories does not contain "
        "memory_text, so missing embeddings cannot be backfilled safely."
    )
    return 0


async def run_episodic_embedder_loop() -> None:
    """Disabled loop placeholder.

    This function intentionally does not run a background embedding loop.
    Remove any scheduling of this loop from api/main.py until a correct
    fallback design is implemented.
    """
    logger.warning(
        "[EpisodicEmbedder] Disabled: loop should not be started. "
        "Remove scheduling from api/main.py."
    )

    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        logger.info("[EpisodicEmbedder] Disabled loop cancelled")
        raise
