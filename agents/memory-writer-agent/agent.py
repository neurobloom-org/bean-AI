"""
BEAN AI — Memory Writer Agent (Privacy-Safe)

After every completed conversation turn:
  1. Extracts structured facts using Gemini Flash
  2. Merges new facts into the user's semantic profile in Supabase
  3. Stores ONLY the vector embedding of the turn (no source text)

Privacy:
  ✓ Raw transcript text is NOT stored to episodic_memories
  ✓ Text sent to Gemini is truncated
  ✓ Text is redacted for obvious PII before LLM submission
  ✓ Episodic memory stores vector only
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types as adk_types

from services.embedding_service import get_embedding
from services.llm_service import generate_json
from services.privacy_service import privacy_service
from services.supabase_client import (
    get_service_client,
    get_user_profile,
    upsert_user_profile,
)

logger = logging.getLogger(__name__)

EXTRACT_FACTS_SYSTEM = """You are a privacy-safe fact extractor for a mental health companion system.

Extract ONLY objective, durable, privacy-safe facts from the conversation snippet.

Rules:
- Extract facts useful across sessions
- Do NOT infer diagnoses
- Do NOT include verbatim quotes
- Do NOT include highly sensitive private details unless plainly necessary
- Return ONLY valid JSON

Return exactly:
{
  "name": "string or null",
  "preferred_name": "string or null",
  "new_interests": ["string"],
  "new_important_people": ["string"],
  "new_personality_notes": ["string"],
  "significant_event": "string or null"
}
"""

EXTRACT_FACTS_PROMPT = """Conversation snippet:

User said (truncated/redacted): {user_text}
Assistant said (truncated/redacted): {assistant_text}
Detected emotion: {emotion}

Existing profile summary:
{existing_profile}

Extract only NEW facts not already present in the profile.
"""


class MemoryWriterAgent(BaseAgent):
    """Writes privacy-safe semantic + episodic memory after each turn."""

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> adk_types.AsyncGenerator:
        state = ctx.session.state

        user_id = state.get("user_id")
        session_id = state.get("session_id")
        user_text = state.get("current_transcript", "")
        assistant_text = state.get("response_text", "")
        emotion = state.get("current_emotion", "neutral")

        if not user_id:
            logger.debug("MemoryWriter: skipped — missing user_id")
            state["memory_write_done"] = "skipped"
            yield adk_types.Content(parts=[])
            return

        if not user_text and not assistant_text:
            logger.debug("MemoryWriter: skipped — empty turn content")
            state["memory_write_done"] = "skipped"
            yield adk_types.Content(parts=[])
            return

        try:
            results = await asyncio.gather(
                self._update_semantic_profile(
                    user_id=user_id,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    emotion=emotion,
                ),
                self._store_episodic_embedding(
                    user_id=user_id,
                    session_id=session_id,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    emotion=emotion,
                ),
                return_exceptions=True,
            )

            had_error = False
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    had_error = True
                    logger.error("MemoryWriter task %d failed: %s", idx, result)

            state["memory_write_done"] = "error" if had_error else "true"

        except Exception as exc:
            logger.exception("MemoryWriter: unexpected failure: %s", exc)
            state["memory_write_done"] = "error"

        yield adk_types.Content(parts=[])

    async def _update_semantic_profile(
        self,
        user_id: str,
        user_text: str,
        assistant_text: str,
        emotion: str,
    ) -> None:
        profile = await get_user_profile(user_id)
        if not profile:
            profile = {
                "user_id": user_id,
                "interests": [],
                "important_people": [],
                "personality_notes": [],
            }

        existing_summary = self._summarise_profile(profile)

        safe_user_text = privacy_service.redact_sensitive_patterns(
            privacy_service.truncate_for_prompt(user_text, max_chars=500)
        )
        safe_assistant_text = privacy_service.redact_sensitive_patterns(
            privacy_service.truncate_for_prompt(assistant_text, max_chars=300)
        )

        prompt = EXTRACT_FACTS_PROMPT.format(
            user_text=safe_user_text or "(empty)",
            assistant_text=safe_assistant_text or "(empty)",
            emotion=emotion or "neutral",
            existing_profile=existing_summary,
        )

        try:
            raw_facts = await generate_json(
                task="fact_extraction",
                prompt=prompt,
                system=EXTRACT_FACTS_SYSTEM,
            )
        except ValueError:
            logger.warning("MemoryWriter: fact extraction returned invalid JSON")
            return

        facts = self._normalise_facts(raw_facts)
        updates = self._build_profile_updates(profile, facts)

        if updates:
            await upsert_user_profile(user_id, updates)
            logger.debug("MemoryWriter: semantic profile updated for user %s", user_id)

    def _summarise_profile(self, profile: dict[str, Any]) -> str:
        parts: list[str] = []

        if profile.get("display_name"):
            parts.append(f"Name: {profile['display_name']}")
        if profile.get("preferred_name"):
            parts.append(f"Preferred name: {profile['preferred_name']}")
        if profile.get("interests"):
            parts.append(f"Interests: {', '.join(profile['interests'][:10])}")
        if profile.get("important_people"):
            parts.append(
                f"Important people: {', '.join(profile['important_people'][:10])}"
            )
        if profile.get("personality_notes"):
            parts.append(f"Notes: {'; '.join(profile['personality_notes'][:5])}")

        return "\n".join(parts) if parts else "No profile yet."

    def _normalise_facts(self, facts: dict[str, Any]) -> dict[str, Any]:
        def clean_string(value: Any, max_len: int = 120) -> str | None:
            if not isinstance(value, str):
                return None
            value = " ".join(value.strip().split())
            if not value:
                return None
            return value[:max_len]

        def clean_list(value: Any, max_items: int = 10, max_len: int = 80) -> list[str]:
            if not isinstance(value, list):
                return []

            cleaned: list[str] = []
            seen: set[str] = set()

            for item in value:
                text = clean_string(item, max_len=max_len)
                if not text:
                    continue
                key = text.casefold()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(text)
                if len(cleaned) >= max_items:
                    break

            return cleaned

        return {
            "name": clean_string(facts.get("name")),
            "preferred_name": clean_string(facts.get("preferred_name")),
            "new_interests": clean_list(facts.get("new_interests")),
            "new_important_people": clean_list(facts.get("new_important_people")),
            "new_personality_notes": clean_list(
                facts.get("new_personality_notes"),
                max_items=10,
                max_len=140,
            ),
            "significant_event": clean_string(
                facts.get("significant_event"),
                max_len=180,
            ),
        }

    def _build_profile_updates(
        self,
        profile: dict[str, Any],
        facts: dict[str, Any],
    ) -> dict[str, Any]:
        updates: dict[str, Any] = {}

        if facts["name"] and not profile.get("display_name"):
            updates["display_name"] = facts["name"]

        if facts["preferred_name"] and not profile.get("preferred_name"):
            updates["preferred_name"] = facts["preferred_name"]

        merged_interests = self._merge_unique_strings(
            profile.get("interests") or [],
            facts.get("new_interests") or [],
        )
        if merged_interests != (profile.get("interests") or []):
            updates["interests"] = merged_interests

        merged_people = self._merge_unique_strings(
            profile.get("important_people") or [],
            facts.get("new_important_people") or [],
        )
        if merged_people != (profile.get("important_people") or []):
            updates["important_people"] = merged_people

        notes_to_add = list(facts.get("new_personality_notes") or [])

        if facts.get("significant_event"):
            timestamp = datetime.now(UTC).strftime("%b %d")
            notes_to_add.append(f"[{timestamp}] {facts['significant_event']}")

        merged_notes = self._merge_unique_strings(
            profile.get("personality_notes") or [],
            notes_to_add,
        )
        if merged_notes != (profile.get("personality_notes") or []):
            updates["personality_notes"] = merged_notes

        if updates:
            updates["last_updated"] = datetime.now(UTC).isoformat()

        return updates

    def _merge_unique_strings(
        self,
        existing_items: list[str],
        new_items: list[str],
        max_items: int = 20,
    ) -> list[str]:
        merged = list(existing_items)
        seen = {item.casefold() for item in existing_items if isinstance(item, str)}

        for item in new_items:
            if not isinstance(item, str):
                continue
            cleaned = " ".join(item.strip().split())
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
            if len(merged) >= max_items:
                break

        return merged

    async def _store_episodic_embedding(
        self,
        user_id: str,
        session_id: str | None,
        user_text: str,
        assistant_text: str,
        emotion: str,
    ) -> None:
        safe_user = privacy_service.redact_sensitive_patterns(
            privacy_service.truncate_for_prompt(user_text, max_chars=300)
        )
        safe_assistant = privacy_service.redact_sensitive_patterns(
            privacy_service.truncate_for_prompt(assistant_text, max_chars=200)
        )

        turn_summary = (
            f"User: {safe_user or '(empty)'}\n"
            f"Assistant: {safe_assistant or '(empty)'}\n"
            f"Emotion: {emotion or 'neutral'}"
        )

        embedding = await get_embedding(turn_summary)

        client = get_service_client()
        client.table("episodic_memories").insert(
            {
                "user_id": user_id,
                "session_id": session_id,
                "embedding": embedding,
                "emotion_label": emotion or "neutral",
                "memory_type": "episodic",
                "source_text": None,
            }
        ).execute()

        logger.debug("MemoryWriter: episodic vector stored for user %s", user_id)


memory_writer_agent = MemoryWriterAgent(name="memory_writer_agent")
