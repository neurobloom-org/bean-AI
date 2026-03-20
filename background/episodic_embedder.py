"""BEAN AI — Episodic embedder background job.

Called periodically to embed any session transcripts that haven't been
embedded yet. In normal flow the MemoryWriterAgent handles this inline,
but this job catches any that were missed.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from shared.config import get_settings

logger = logging.getLogger(__name__)


async def embed_unprocessed_sessions() -> int:
    """Find recently-ended sessions without episodic embeddings and embed them."""
    from services.embedding_service import get_embedding
    from services.privacy_service import privacy_service
    from services.supabase_client import get_service_client

    settings = get_settings()
    client = await get_service_client()
    embedded_count = 0

    try:
        cutoff = (datetime.now(UTC) - timedelta(hours=2)).isoformat()

        result = (
            await client.table("sessions")
            .select("id, user_id, ended_at")
            .eq("status", "ended")
            .gte("ended_at", cutoff)
            .order("ended_at", desc=False)
            .execute()
        )

        for session in (result.data or []):
            session_id = session["id"]
            user_id = session["user_id"]

            existing = (
                await client.table("episodic_memories")
                .select("id")
                .eq("session_id", session_id)
                .limit(1)
                .execute()
            )
            if existing.data:
                continue

            turns = await privacy_service.get_recent_transcript(
                session_id=session_id,
                max_turns=20,
            )
            if not turns:
                continue

            user_turns = [
                t.get("text", "").strip()
                for t in turns
                if t.get("speaker") == "user" and t.get("text")
            ]
            if not user_turns:
                continue

            summary = " ".join(user_turns[:5])[:500]
            if not summary.strip():
                continue

            try:
                embedding = await get_embedding(summary)

                expires_at = (
                    datetime.now(UTC)
                    + timedelta(days=settings.episodic_memory_retention_days)
                ).isoformat()

                await (
                    client.table("episodic_memories")
                    .insert(
                        {
                            "user_id": user_id,
                            "session_id": session_id,
                            "embedding": embedding,
                            "memory_type": "session_summary",
                            "expires_at": expires_at,
                        }
                    )
                    .execute()
                )

                embedded_count += 1
                logger.info(
                    "[EpisodicEmbedder] Embedded session=%s user=%s",
                    str(session_id)[:8],
                    str(user_id)[:8],
                )

            except Exception as exc:
                logger.error(
                    "[EpisodicEmbedder] Embedding failed for session %s: %s",
                    str(session_id)[:8],
                    exc,
                )

    except Exception as exc:
        logger.error("[EpisodicEmbedder] Job failed: %s", exc)

    return embedded_count


async def run_episodic_embedder_loop() -> None:
    """Run episodic embedding every 30 minutes."""
    interval_seconds = 1800
    logger.info("[EpisodicEmbedder] Loop started — interval=30m")

    while True:
        try:
            count = await embed_unprocessed_sessions()
            if count:
                logger.info("[EpisodicEmbedder] Embedded %d sessions", count)

            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            logger.info("[EpisodicEmbedder] Loop cancelled")
            break
        except Exception as exc:
            logger.error("[EpisodicEmbedder] Loop error: %s", exc)
            await asyncio.sleep(interval_seconds)