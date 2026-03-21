import asyncio
import logging
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from services.embedding_service import search_similar_memories
from services.supabase_client import get_user_profile
from shared.schemas import (
    EpisodicMemoryResult,
    MemoryContext,
    UserProfile,
    WorkingMemoryEntry,
)

logger = logging.getLogger(__name__)


class MemoryAgent(BaseAgent):
    """Retrieve memory context from Supabase-backed sources in parallel."""

    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        session_state = ctx.session.state

        user_id = str(session_state.get("user_id", "") or "").strip()
        session_id = str(session_state.get("session_id", "") or "").strip()
        current_text = str(session_state.get("current_transcript", "") or "").strip()

        if not user_id:
            logger.warning("MemoryAgent: missing user_id; returning graceful fallback")
            fallback = "No memory context yet."
            session_state["memory_context"] = fallback
            yield self._build_state_event(fallback)
            return

        # Run all three reads in parallel. Each helper gracefully degrades.
        working_memory, user_profile, episodic_memories = await asyncio.gather(
            self._get_working_memory(session_id=session_id),
            self._get_user_profile(user_id=user_id),
            self._get_episodic_memories(
                user_id=user_id,
                query_text=current_text,
            ),
        )

        # If there is no current transcript, try using the recent working memory text
        # as a fallback query for episodic search.
        if not episodic_memories and not current_text and working_memory:
            fallback_query = " ".join(
                entry.text.strip()
                for entry in working_memory[-3:]
                if getattr(entry, "text", "").strip()
            )[:300]

            if fallback_query:
                episodic_memories = await self._get_episodic_memories(
                    user_id=user_id,
                    query_text=fallback_query,
                )

        memory_context_string = self._assemble_memory_context(
            working_memory=working_memory,
            user_profile=user_profile,
            episodic_memories=episodic_memories,
        )

        session_state["memory_context"] = memory_context_string

        logger.info(
            "MemoryAgent assembled context: working=%d profile=%s episodic=%d",
            len(working_memory),
            "yes" if user_profile else "no",
            len(episodic_memories),
        )

        yield self._build_state_event(memory_context_string)

    def _build_state_event(self, memory_context: str) -> Event:
        """Build a session state update event."""
        return Event(
            author=self.name,
            actions={
                "state_delta": {
                    "memory_context": memory_context,
                }
            },
        )

    def _assemble_memory_context(
        self,
        working_memory: list[WorkingMemoryEntry],
        user_profile: UserProfile | None,
        episodic_memories: list[EpisodicMemoryResult],
    ) -> str:
        """
        Assemble a MemoryContext and convert it to prompt string.

        Returns a stable fallback string when all sources are empty.
        """
        if not working_memory and not user_profile and not episodic_memories:
            return "No memory context yet."

        memory_ctx = MemoryContext(
            working_memory=working_memory,
            user_profile=user_profile,
            episodic_memories=episodic_memories,
        )

        try:
            prompt_string = memory_ctx.to_prompt_string()
            if isinstance(prompt_string, str) and prompt_string.strip():
                return prompt_string.strip()
        except Exception as exc:
            logger.error("MemoryContext.to_prompt_string() failed: %s", exc)

        return "No memory context yet."

    async def _get_working_memory(self, session_id: str) -> list[WorkingMemoryEntry]:
        """Fetch the last 6 transcript turns for the session."""
        if not session_id:
            logger.warning("MemoryAgent: missing session_id; working memory skipped")
            return []

        try:
            from services.privacy_service import privacy_service

            turns = await privacy_service.get_recent_transcript(
                session_id=session_id,
                max_turns=6,
            )

            results: list[WorkingMemoryEntry] = []
            for turn in turns or []:
                speaker = str(turn.get("speaker", "") or "").strip()
                text = str(turn.get("text", "") or "").strip()

                if not speaker or not text:
                    continue

                results.append(
                    WorkingMemoryEntry(
                        speaker=speaker,
                        text=text,
                    )
                )

            return results

        except Exception as exc:
            logger.error("MemoryAgent: working memory retrieval failed: %s", exc)
            return []

    async def _get_user_profile(self, user_id: str) -> UserProfile | None:
        """Fetch semantic profile from user_profiles."""
        try:
            profile_data = await get_user_profile(user_id)

            if not profile_data:
                return None

            return UserProfile(**profile_data)

        except Exception as exc:
            logger.error("MemoryAgent: user profile retrieval failed: %s", exc)
            return None

    async def _get_episodic_memories(
        self,
        user_id: str,
        query_text: str,
        limit: int = 3,
    ) -> list[EpisodicMemoryResult]:
        """Fetch top episodic memories using vector similarity search."""
        cleaned_query = (query_text or "").strip()
        if not cleaned_query:
            return []

        try:
            memories = await search_similar_memories(
                user_id=user_id,
                query_text=cleaned_query[:300],
                top_k=limit,
                min_similarity=0.72,
            )

            results: list[EpisodicMemoryResult] = []
            for memory in memories or []:
                similarity = float(memory.get("similarity", 0.0) or 0.0)

                # Keep aligned with search threshold; avoid weaker matches.
                if similarity < 0.72:
                    continue

                results.append(
                    EpisodicMemoryResult(
                        memory_id=str(memory.get("id", "") or ""),
                        emotion_label=memory.get("emotion_label"),
                        similarity_score=similarity,
                        created_at=memory.get("created_at"),
                    )
                )

            return results

        except Exception as exc:
            logger.error("MemoryAgent: episodic memory retrieval failed: %s", exc)
            return []


memory_agent = MemoryAgent(name="memory_agent")