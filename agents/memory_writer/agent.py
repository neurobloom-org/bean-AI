"""BEAN AI — Memory Writer Agent (Privacy-Safe)

After every completed conversation turn:
  1. Extracts structured facts from the current turn via LLM
  2. Fetches existing profile, merges new facts, and upserts the result
  3. Stores ONLY the vector embedding of the turn (never raw source text)

Privacy guarantees:
  ✓ Raw transcript text is NEVER stored in episodic_memories
  ✓ Text sent to the LLM is PII-redacted first, then truncated
  ✓ Episodic rows contain only the embedding vector + metadata
  ✓ source_text is excluded from all DB inserts — enforced by code, not assertion

Called by the orchestrator as fire-and-forget via asyncio.create_task().
Writes session.state["memory_write_done"] = "true" | "error" | "skipped".

Design note — async DB calls:
    All Supabase calls use AsyncClient.execute(), which is a coroutine.
    They must be awaited directly in async functions, never passed to
    asyncio.to_thread(). _retry_sync_in_thread has been removed entirely;
    all DB retries go through _retry_async with proper async callables.

FIX: Removed the duplicate LLM-compress episodic store block that appeared
AFTER the asyncio.gather() in _run_async_impl. The gather already calls
self._store_episodic_embedding() which writes to episodic_memories. The
duplicate block below it wrote a SECOND row to the same table for every
turn, using a second embedding API call and generating duplicate vector
search results. The duplication was introduced when the LLM-summary path
was added without noticing that _store_episodic_embedding was already
handling episodic storage.
"""

from __future__ import annotations

import asyncio
import logging
from asyncio import timeout as async_timeout
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from services.embedding_service import get_embedding
from services.llm_service import generate, generate_json
from services.privacy_service import privacy_service
from services.supabase_client import get_service_client
from shared.config import get_settings
from shared.exceptions import EmbeddingError, LLMError, SupabaseError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── Tuneable constants ────────────────────────────────────────────────────────

_GATHER_TIMEOUT_S: float = 20.0

_USER_FACT_TEXT_MAX_CHARS: int = 500
_ASSISTANT_FACT_TEXT_MAX_CHARS: int = 300
_USER_EMBED_TEXT_MAX_CHARS: int = 300
_ASSISTANT_EMBED_TEXT_MAX_CHARS: int = 200

_EMOTION_MAX_LEN: int = 40

_EXPECTED_EMBEDDING_DIMENSIONS: int = 1536

_DB_MAX_ATTEMPTS: int = 3
_DB_RETRY_BASE_DELAY_S: float = 0.20

_LLM_MAX_ATTEMPTS: int = 2
_LLM_RETRY_BASE_DELAY_S: float = 0.35

# ── LLM prompt templates ─────────────────────────────────────────────────────

_EXTRACT_FACTS_SYSTEM = """\
You are a privacy-safe fact extractor for a mental health companion system.

Extract ONLY objective, durable, privacy-safe facts from the CURRENT conversation turn.

Rules:
- Extract facts useful across sessions
- Do NOT infer diagnoses
- Do NOT include verbatim quotes
- Do NOT include highly sensitive private details unless plainly necessary
- Prefer facts directly stated by the user
- Do NOT try to deduplicate against any prior profile; the database merge layer handles that
- Return ONLY valid JSON — no markdown fences, no extra keys
"""

_EXTRACT_FACTS_PROMPT = """\
Current conversation turn:

User said (redacted/truncated): {user_text}
Assistant said (redacted/truncated): {assistant_text}
Detected emotion: {emotion}

Extract durable, privacy-safe candidate facts from this turn only.
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _clean_optional_string(value: Any, max_len: int = 120) -> str | None:
    """Return a normalized bounded string or None."""
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return None
    return cleaned[:max_len]


def _coerce_string_list(
    value: Any,
    *,
    max_items: int = 50,
    max_item_len: int = 120,
) -> list[str]:
    """Safely coerce an unknown value into a deduplicated normalized string list."""
    if not isinstance(value, list):
        return []

    cleaned_items: list[str] = []
    seen: set[str] = set()

    for item in value:
        cleaned = _clean_optional_string(item, max_item_len)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned_items.append(cleaned)
        if len(cleaned_items) >= max_items:
            break

    return cleaned_items


def _merge_string_lists(existing: list[str], new_items: list[str]) -> list[str]:
    """Append new_items to existing, skipping case-insensitive duplicates.

    Preserves original casing of first-seen items. Existing items always
    come first so established profile data is not reordered.
    """
    seen = {item.casefold() for item in existing}
    merged = list(existing)
    for item in new_items:
        if item.casefold() not in seen:
            seen.add(item.casefold())
            merged.append(item)
    return merged


# ── Pydantic models ───────────────────────────────────────────────────────────


class ExtractedFacts(BaseModel):
    """Typed fact payload returned by the LLM.

    extra="ignore" rather than "forbid": the LLM may include reasoning fields
    or other extra keys. Forbidding them causes ValidationError and silently
    drops the entire fact extraction result for the turn.
    """

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    preferred_name: str | None = None
    new_interests: list[str] = Field(default_factory=list)
    new_important_people: list[str] = Field(default_factory=list)
    new_personality_notes: list[str] = Field(default_factory=list)
    significant_event: str | None = None

    @field_validator("name", "preferred_name", mode="before")
    @classmethod
    def validate_names(cls, value: Any) -> str | None:
        return _clean_optional_string(value, 120)

    @field_validator("significant_event", mode="before")
    @classmethod
    def validate_significant_event(cls, value: Any) -> str | None:
        return _clean_optional_string(value, 180)

    @field_validator("new_interests", "new_important_people", mode="before")
    @classmethod
    def validate_short_lists(cls, value: Any) -> list[str]:
        return _coerce_string_list(value, max_items=10, max_item_len=80)

    @field_validator("new_personality_notes", mode="before")
    @classmethod
    def validate_long_lists(cls, value: Any) -> list[str]:
        return _coerce_string_list(value, max_items=10, max_item_len=140)


# ── Agent ─────────────────────────────────────────────────────────────────────


class MemoryWriterAgent(BaseAgent):
    """Writes privacy-safe semantic + episodic memory after each conversation turn."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        user_id_raw = state.get("user_id")
        session_id_raw = state.get("session_id")
        user_text = str(state.get("current_transcript") or "")
        assistant_text = str(state.get("response_text") or "")
        raw_emotion = str(state.get("current_emotion") or "neutral")

        user_id = str(user_id_raw).strip() if user_id_raw else None
        session_id = str(session_id_raw).strip() if session_id_raw else None

        if not user_id:
            logger.debug("MemoryWriter: skipped — missing user_id")
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"memory_write_done": "skipped"}),
            )
            return

        if not user_text.strip() and not assistant_text.strip():
            logger.debug("MemoryWriter: skipped — empty turn content")
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"memory_write_done": "skipped"}),
            )
            return

        emotion = _sanitize_emotion(raw_emotion)

        try:
            async with async_timeout(_GATHER_TIMEOUT_S):
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
        except TimeoutError:
            logger.error(
                "MemoryWriter: timed out after %.1fs for user %s",
                _GATHER_TIMEOUT_S,
                user_id,
            )
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"memory_write_done": "error"}),
            )
            return
        except asyncio.CancelledError:
            logger.warning("MemoryWriter: cancelled for user %s", user_id)
            raise

        had_error = False
        task_names = ("semantic_profile", "episodic_embedding")

        for task_name, result in zip(task_names, results, strict=True):
            if isinstance(result, Exception):
                had_error = True
                logger.error(
                    "MemoryWriter: task '%s' failed for user %s",
                    task_name,
                    user_id,
                    exc_info=(type(result), result, result.__traceback__),
                )
            elif result is not True:
                had_error = True
                logger.warning(
                    "MemoryWriter: task '%s' reported unsuccessful completion for user %s",
                    task_name,
                    user_id,
                )

        # FIX: Removed the duplicate LLM-compress episodic store block that
        # was here in the original code. It wrote a second row to episodic_memories
        # for every turn, despite _store_episodic_embedding() (called in the
        # gather above) already handling episodic storage. This caused:
        #   1. Two episodic_memories rows per turn (duplicate vectors)
        #   2. Two embedding API calls per turn (wasted cost)
        #   3. Duplicate results in similarity searches
        #
        # The privacy model is preserved: _store_episodic_embedding stores a
        # redacted turn summary as the embedding source without persisting the
        # source text itself. Downstream code that needs to display episodic
        # context (therapeutic_convo/_fetch_episodic_memories) can use the
        # metadata fields (emotion_label, created_at) rather than raw content.

        # Task tracking: clear flags set by sub-agents that have already done
        # their own persistence.

        # task_just_saved_id: already written to tasks table by task_agent._save_task
        task_id = state.get("task_just_saved_id")
        if task_id:
            ctx.session.state["task_just_saved_id"] = None

        # Safety alert: already written to alerts table by alert_agent._dispatch_alert

        memory_write_done = "error" if had_error else "true"

        if not had_error:
            logger.info("MemoryWriter: both tasks succeeded for user %s", user_id)

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={"memory_write_done": memory_write_done}),
        )

    # ── Semantic profile ──────────────────────────────────────────────────────

    async def _update_semantic_profile(
        self,
        user_id: str,
        user_text: str,
        assistant_text: str,
        emotion: str,
    ) -> bool:
        """Extract facts from the current turn and merge them into the user profile."""
        try:
            safe_user = _safe_redact_and_truncate(user_text, _USER_FACT_TEXT_MAX_CHARS)
            safe_assistant = _safe_redact_and_truncate(
                assistant_text,
                _ASSISTANT_FACT_TEXT_MAX_CHARS,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "MemoryWriter: unexpected privacy preparation failure in semantic path"
            )
            return False

        prompt = _EXTRACT_FACTS_PROMPT.format(
            user_text=safe_user or "(empty)",
            assistant_text=safe_assistant or "(empty)",
            emotion=emotion,
        )

        # Step 1 — extract facts via LLM.
        try:
            raw_facts = await _retry_async(
                lambda: generate_json(
                    task="fact_extraction",
                    prompt=prompt,
                    system=_EXTRACT_FACTS_SYSTEM,
                ),
                max_attempts=_LLM_MAX_ATTEMPTS,
                base_delay_s=_LLM_RETRY_BASE_DELAY_S,
                operation_name="fact extraction",
                retriable_exceptions=(LLMError, ValueError),
            )
        except asyncio.CancelledError:
            raise
        except (LLMError, ValueError) as exc:
            logger.warning(
                "MemoryWriter: fact extraction failed for user %s: %s",
                user_id,
                exc,
            )
            return False
        except Exception:
            logger.exception(
                "MemoryWriter: unexpected fact extraction error for user %s",
                user_id,
            )
            return False

        try:
            facts = _parse_extracted_facts(raw_facts)
        except ValidationError as exc:
            logger.warning(
                "MemoryWriter: fact schema validation failed for user %s: %s",
                user_id,
                exc,
            )
            return False

        # Step 2 — fetch existing profile so we can merge arrays, not overwrite.
        existing_profile: dict[str, Any] = {}
        try:

            async def _fetch_profile() -> dict[str, Any]:
                client = await get_service_client()
                result = (
                    await client.table("user_profiles")
                    .select(
                        "display_name, preferred_name, interests, "
                        "important_people, personality_notes"
                    )
                    .eq("user_id", user_id)
                    .maybe_single()
                    .execute()
                )
                return dict(result.data) if result and result.data else {}

            existing_profile = await _retry_async(
                _fetch_profile,
                max_attempts=_DB_MAX_ATTEMPTS,
                base_delay_s=_DB_RETRY_BASE_DELAY_S,
                operation_name="profile fetch",
                retriable_exceptions=(Exception,),
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "MemoryWriter: could not fetch existing profile for user %s — "
                "writing facts without merge (arrays may be narrowed this turn)",
                user_id,
            )

        # Step 3 — build the merged payload.
        merged_payload = self._build_merged_profile_payload(
            facts=facts,
            existing=existing_profile,
        )

        if not merged_payload:
            logger.debug(
                "MemoryWriter: no new semantic facts to persist for user %s",
                user_id,
            )
            return True

        # Step 4 — upsert the merged result.
        try:

            async def _do_upsert() -> None:
                client = await get_service_client()
                await (
                    client.table("user_profiles")
                    .upsert(
                        {"user_id": user_id, **merged_payload},
                        on_conflict="user_id",
                    )
                    .execute()
                )

            await _retry_async(
                _do_upsert,
                max_attempts=_DB_MAX_ATTEMPTS,
                base_delay_s=_DB_RETRY_BASE_DELAY_S,
                operation_name="semantic profile upsert",
                retriable_exceptions=(Exception,),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise SupabaseError(
                f"MemoryWriter: failed to upsert semantic profile for user {user_id}"
            ) from exc

        logger.info(
            "MemoryWriter: semantic profile merged for user %s (keys: %s)",
            user_id,
            sorted(merged_payload.keys()),
        )
        return True

    # ── Episodic embedding ────────────────────────────────────────────────────

    async def _store_episodic_embedding(
        self,
        user_id: str,
        session_id: str | None,
        user_text: str,
        assistant_text: str,
        emotion: str,
    ) -> bool:
        """Embed a redacted turn summary and store the vector — never the text.

        This is the ONLY episodic memory write per turn. The source text is
        redacted and truncated before embedding, and is NOT stored in the DB row.
        Only the vector and metadata (emotion_label, session_id, expires_at) are
        persisted, maintaining the privacy guarantee.
        """
        try:
            safe_user = _safe_redact_and_truncate(user_text, _USER_EMBED_TEXT_MAX_CHARS)
            safe_assistant = _safe_redact_and_truncate(
                assistant_text,
                _ASSISTANT_EMBED_TEXT_MAX_CHARS,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "MemoryWriter: unexpected privacy preparation failure in episodic path"
            )
            return False

        if not safe_user.strip() and not safe_assistant.strip():
            logger.debug(
                "MemoryWriter: episodic embedding skipped — turn empty after redaction"
            )
            return True

        turn_summary = (
            f"User: {safe_user or '(empty)'}\n"
            f"Assistant: {safe_assistant or '(empty)'}\n"
            f"Emotion: {emotion}"
        )

        # Step 1 — generate embedding.
        try:
            embedding = await _retry_async(
                lambda: get_embedding(turn_summary),
                max_attempts=_DB_MAX_ATTEMPTS,
                base_delay_s=_DB_RETRY_BASE_DELAY_S,
                operation_name="embedding generation",
                retriable_exceptions=(EmbeddingError, Exception),
            )
        except asyncio.CancelledError:
            raise
        except EmbeddingError as exc:
            raise EmbeddingError(
                f"MemoryWriter: embedding failed for user {user_id}"
            ) from exc
        except Exception as exc:
            raise EmbeddingError(
                f"MemoryWriter: unexpected embedding error for user {user_id}"
            ) from exc

        if not _is_valid_embedding(
            embedding,
            expected_dimensions=_EXPECTED_EMBEDDING_DIMENSIONS,
        ):
            raise EmbeddingError(
                f"MemoryWriter: invalid embedding payload for user {user_id}; "
                f"expected {_EXPECTED_EMBEDDING_DIMENSIONS} dimensions, "
                f"got {len(embedding) if isinstance(embedding, list) else type(embedding).__name__}"
            )

        settings = get_settings()
        expires_at = (
            datetime.now(UTC) + timedelta(days=settings.episodic_memory_retention_days)
        ).isoformat()

        # NOTE: source_text is intentionally NOT stored in this row.
        # Only the vector + metadata is persisted (privacy guarantee).
        row: dict[str, Any] = {
            "user_id": user_id,
            "session_id": session_id,
            "embedding": embedding,
            "emotion_label": emotion,
            "memory_type": "episodic",
            "expires_at": expires_at,
        }

        # Step 2 — insert the vector row.
        try:

            async def _do_insert() -> None:
                client = await get_service_client()
                await client.table("episodic_memories").insert(row).execute()

            await _retry_async(
                _do_insert,
                max_attempts=_DB_MAX_ATTEMPTS,
                base_delay_s=_DB_RETRY_BASE_DELAY_S,
                operation_name="episodic insert",
                retriable_exceptions=(Exception,),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise SupabaseError(
                f"MemoryWriter: failed to insert episodic row for user {user_id}"
            ) from exc

        logger.info(
            "MemoryWriter: episodic vector stored for user %s (session %s, expires %s)",
            user_id,
            session_id or "none",
            expires_at[:10],
        )
        return True

    # ── Profile payload builder ───────────────────────────────────────────────

    def _build_merged_profile_payload(
        self,
        facts: ExtractedFacts,
        existing: dict[str, Any],
    ) -> dict[str, Any]:
        """Build a merged upsert payload from extracted facts + existing profile."""
        payload: dict[str, Any] = {}

        if facts.name:
            payload["display_name"] = facts.name

        if facts.preferred_name:
            payload["preferred_name"] = facts.preferred_name

        if facts.new_interests:
            existing_interests = existing.get("interests") or []
            if isinstance(existing_interests, list):
                payload["interests"] = _merge_string_lists(
                    existing_interests, facts.new_interests
                )
            else:
                payload["interests"] = facts.new_interests

        if facts.new_important_people:
            existing_people = existing.get("important_people") or []
            if isinstance(existing_people, list):
                payload["important_people"] = _merge_string_lists(
                    existing_people, facts.new_important_people
                )
            else:
                payload["important_people"] = facts.new_important_people

        notes_to_add = list(facts.new_personality_notes)
        if facts.significant_event:
            timestamp = datetime.now(UTC).strftime("%b %d %Y")
            notes_to_add.append(f"[{timestamp}] {facts.significant_event}")

        if notes_to_add:
            existing_notes = existing.get("personality_notes") or []
            if isinstance(existing_notes, list):
                payload["personality_notes"] = _merge_string_lists(
                    existing_notes, notes_to_add
                )
            else:
                payload["personality_notes"] = notes_to_add

        if payload:
            payload["last_updated"] = datetime.now(UTC).isoformat()

        return payload


# ── Module-level helpers ──────────────────────────────────────────────────────


def _sanitize_emotion(raw: str) -> str:
    """Clamp the emotion string to a safe length and strip unsafe chars."""
    cleaned = "".join(c for c in str(raw).lower() if c.isalnum() or c in {"_", " "})
    cleaned = " ".join(cleaned.split())[:_EMOTION_MAX_LEN]
    return cleaned or "neutral"


def _safe_redact_and_truncate(text: str, max_chars: int) -> str:
    """Redact PII first, then truncate, with strict type safety."""
    try:
        redacted = privacy_service.redact_sensitive_patterns(text)

        if not isinstance(redacted, str):
            logger.warning(
                "MemoryWriter: privacy_service returned non-string type %s; "
                "using empty string fallback",
                type(redacted).__name__,
            )
            return ""

        truncated = privacy_service.truncate_for_prompt(redacted, max_chars=max_chars)

        if not isinstance(truncated, str):
            logger.warning(
                "MemoryWriter: truncation returned non-string type %s; "
                "using empty string fallback",
                type(truncated).__name__,
            )
            return ""

        return truncated.strip()

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.warning(
            "MemoryWriter: privacy_service failed (%s); using empty string fallback",
            exc,
        )
        return ""


def _parse_extracted_facts(raw_facts: Any) -> ExtractedFacts:
    """Parse LLM output whether the wrapper returns a dict or a raw JSON string."""
    if isinstance(raw_facts, str):
        return ExtractedFacts.model_validate_json(raw_facts)
    return ExtractedFacts.model_validate(raw_facts)


def _is_valid_embedding(
    value: Any,
    *,
    expected_dimensions: int,
) -> bool:
    """Sanity-check embedding payload shape and dimensionality."""
    if not isinstance(value, list):
        return False
    if len(value) != expected_dimensions:
        return False
    return isinstance(value[0], (int, float)) and isinstance(value[-1], (int, float))


async def _retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    base_delay_s: float,
    operation_name: str,
    retriable_exceptions: tuple[type[BaseException], ...],
) -> T:
    """Retry transient async operations with exponential backoff."""
    last_exc: BaseException | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return await func()
        except asyncio.CancelledError:
            raise
        except retriable_exceptions as exc:
            last_exc = exc
            if attempt >= max_attempts:
                break
            delay = base_delay_s * (2 ** (attempt - 1))
            logger.warning(
                "MemoryWriter: %s failed on attempt %d/%d: %s; retrying in %.2fs",
                operation_name,
                attempt,
                max_attempts,
                exc,
                delay,
            )
            await asyncio.sleep(delay)

    if last_exc is None:
        raise RuntimeError(
            f"_retry_async for '{operation_name}' exhausted retries but last_exc is None"
        )
    raise last_exc


# ── Singleton export ──────────────────────────────────────────────────────────

memory_writer_agent = MemoryWriterAgent(name="memory_writer_agent")

__all__ = ["memory_writer_agent", "MemoryWriterAgent", "ExtractedFacts"]