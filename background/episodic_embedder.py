"""BEAN AI — Episodic embedder background job.

Runs every 5 minutes.
Finds episodic_memories rows where embedding is NULL,
generates embeddings, and updates those rows.

Used as a fallback when memory_writer_agent fails.
"""

import asyncio
import logging
from collections.abc import Mapping

from services.embedding_service import get_embedding
from services.supabase_client import get_service_client

logger = logging.getLogger(__name__)


async def embed_missing_episodic_memories() -> int:
    """Generate embeddings for episodic memory rows missing embeddings."""
    try:
        client = await get_service_client()

        result = (
            await client.table("episodic_memories")
            .select("id, memory_text")
            .is_("embedding", "null")
            .limit(50)
            .execute()
        )

        rows = result.data or []
        updated_count = 0

        for row in rows:
            if not isinstance(row, Mapping):
                logger.warning("[EpisodicEmbedder] Skipping non-mapping row")
                continue

            memory_id = row.get("id")
            memory_text = row.get("memory_text")

            if not memory_id or not isinstance(memory_text, str) or not memory_text:
                logger.warning(
                    "[EpisodicEmbedder] Skipping row with missing id or memory_text"
                )
                continue

            try:
                embedding = await get_embedding(memory_text)

                await (
                    client.table("episodic_memories")
                    .update({"embedding": embedding})
                    .eq("id", memory_id)
                    .execute()
                )

                updated_count += 1

                logger.info(
                    "[EpisodicEmbedder] Updated embedding for memory %s",
                    str(memory_id)[:8],
                )

            except Exception as exc:
                logger.error(
                    "[EpisodicEmbedder] Failed to embed memory %s: %s",
                    str(memory_id)[:8],
                    exc,
                )

        return updated_count

    except Exception as exc:
        logger.error("[EpisodicEmbedder] Job failed: %s", exc)
        return 0


async def run_episodic_embedder_loop() -> None:
    """Run episodic embedder every 5 minutes."""
    interval_seconds = 5 * 60

    logger.info("[EpisodicEmbedder] Loop started — interval=5m")

    while True:
        try:
            count = await embed_missing_episodic_memories()

            if count:
                logger.info(
                    "[EpisodicEmbedder] Updated %d missing embeddings",
                    count,
                )

            await asyncio.sleep(interval_seconds)

        except asyncio.CancelledError:
            logger.info("[EpisodicEmbedder] Loop cancelled")
            break

        except Exception as exc:
            logger.error("[EpisodicEmbedder] Loop error: %s", exc)
            await asyncio.sleep(interval_seconds)