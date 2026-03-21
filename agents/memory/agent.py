"""BEAN AI v5 — Memory Agent.

Fetches context from three sources in parallel:
  1. Recent transcript turns    — session_transcripts table (24-h TTL)
  2. Semantic profile           — user_profiles table
  3. Episodic memories          — episodic_memories table (pgvector cosine search)

Assembles a MemoryContext string injected into all response-agent prompts.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import re
import time
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from pydantic import BaseModel, Field, ValidationError, field_validator

from services.embedding_service import search_similar_memories
from services.supabase_client import get_user_profile
from shared.schemas import (
    EpisodicMemoryResult,
    MemoryContext,
    UserProfile,
    WorkingMemoryEntry,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_MIN_SIMILARITY: float = 0.72
_MAX_WORKING_TURNS: int = 6
_MAX_EPISODIC_RESULTS: int = 3
_MAX_QUERY_CHARS: int = 300
_FALLBACK_QUERY_TURNS: int = 3

_WORKING_MEMORY_TIMEOUT_SECONDS: float = 3.0
_USER_PROFILE_TIMEOUT_SECONDS: float = 3.0
_EPISODIC_MEMORY_TIMEOUT_SECONDS: float = 6.0
_FALLBACK_EPISODIC_TIMEOUT_SECONDS: float = 4.0

_RETRY_ATTEMPTS: int = 3
_RETRY_BASE_DELAY_SECONDS: float = 0.35
_RETRY_MAX_DELAY_SECONDS: float = 2.0

# FIX: raised from 10 → 20. Breakers are shared across all concurrent users
# (if Supabase is down, it is down for everyone — that is intentional).
# A threshold of 10 means one unlucky session can trip the breaker for all
# other users. 20 accounts for realistic concurrent traffic.
_CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 20
_CIRCUIT_BREAKER_OPEN_SECONDS: float = 30.0

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WHITESPACE_RE = re.compile(r"\s+")


# ── Dependency Protocols ──────────────────────────────────────────────────────
class PrivacyClientProtocol(Protocol):
    async def get_recent_transcript(
        self,
        *,
        session_id: str,
        max_turns: int,
    ) -> list[dict[str, Any]]: ...


class UserProfileFetcherProtocol(Protocol):
    async def __call__(self, user_id: str) -> dict[str, Any] | None: ...


class EpisodicSearchProtocol(Protocol):
    async def __call__(
        self,
        *,
        user_id: str,
        query_text: str,
        top_k: int,
        min_similarity: float,
    ) -> list[dict[str, Any]]: ...


# ── Boundary Models ───────────────────────────────────────────────────────────
class RawWorkingMemoryRow(BaseModel):
    """Validates a raw DB row before constructing a WorkingMemoryEntry.
    A malformed row raises ValidationError on that row alone, not the whole batch.
    """

    speaker: str
    text: str

    @field_validator("speaker", "text", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("expected string")
        cleaned = _normalize_text(value)
        if not cleaned:
            raise ValueError("empty string")
        return cleaned


class RawEpisodicRow(BaseModel):
    """Validates a raw episodic DB row before constructing an EpisodicMemoryResult.
    Guards against null created_at — non-optional in EpisodicMemoryResult — which
    would otherwise crash the Pydantic constructor and silently drop the whole batch.
    """

    id: str
    emotion_label: str | None = None
    similarity: float = Field(default=0.0)
    created_at: datetime

    @field_validator("id", mode="before")
    @classmethod
    def clean_id(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("id must be a string")
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("id cannot be empty")
        return cleaned

    @field_validator("similarity", mode="before")
    @classmethod
    def clean_similarity(cls, value: Any) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid similarity") from exc
        if math.isnan(parsed) or math.isinf(parsed):
            raise ValueError("similarity cannot be NaN or infinite")
        return parsed

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_created_at(cls, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("created_at cannot be empty")
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("invalid datetime string") from exc
        raise ValueError("created_at must be a datetime or ISO string")


# ── Result Containers ─────────────────────────────────────────────────────────
class SourceResult(BaseModel):
    ok: bool
    timed_out: bool = False
    retries_used: int = 0
    latency_ms: float = 0.0
    item_count: int | None = None
    error_type: str | None = None


@dataclass(slots=True)
class MemoryFetchOutcome:
    working_memory: list[WorkingMemoryEntry]
    working_meta: SourceResult
    user_profile: UserProfile | None
    profile_meta: SourceResult
    episodic_memories: list[EpisodicMemoryResult]
    episodic_meta: SourceResult


# ── Circuit Breaker ───────────────────────────────────────────────────────────
class CircuitBreaker:
    """Minimal, synchronous, lock-free circuit breaker for asyncio services."""

    def __init__(self, *, failure_threshold: int, open_seconds: float) -> None:
        self._failure_threshold = failure_threshold
        self._open_seconds = open_seconds
        self._consecutive_failures = 0
        self._opened_until_monotonic = 0.0

    def allow_request(self) -> bool:
        return time.monotonic() >= self._opened_until_monotonic

    def record_success(self) -> None:
        self._consecutive_failures = 0
        self._opened_until_monotonic = 0.0

    def record_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._failure_threshold:
            self._opened_until_monotonic = time.monotonic() + self._open_seconds

    def snapshot(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "consecutive_failures": self._consecutive_failures,
            "is_open": now < self._opened_until_monotonic,
            "open_remaining_seconds": max(0.0, self._opened_until_monotonic - now),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────
def _normalize_text(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value).strip()


def _truncate_at_word_boundary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_space = truncated.rfind(" ")
    if last_space <= 0:
        return truncated.strip()
    return truncated[:last_space].strip()


def _build_sentence_aware_query(parts: list[str], max_chars: int) -> str:
    text = _normalize_text(" ".join(parts))
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if not sentences:
        return _truncate_at_word_boundary(text, max_chars)

    selected: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        projected = len(sentence) if not selected else total + 1 + len(sentence)
        if projected > max_chars:
            if not selected:
                return _truncate_at_word_boundary(sentence, max_chars)
            break
        selected.append(sentence)
        total = projected

    if not selected:
        return _truncate_at_word_boundary(text, max_chars)

    selected.reverse()
    return " ".join(selected).strip()


# ── Agent ─────────────────────────────────────────────────────────────────────
class MemoryAgent(BaseAgent):
    """Retrieve memory context from Supabase-backed sources in parallel.

    Accepts injected service dependencies so tests can pass mock implementations
    without patching module globals.
    """

    model_config = {"arbitrary_types_allowed": True}

    def __init__(
        self,
        *,
        name: str,
        # FIX: privacy_service removed from top-level import and resolved here
        # via a deferred local import. A top-level import risks a circular import
        # at startup depending on Python's module initialisation order.
        # Passing a mock in tests still works — just pass privacy_client= directly.
        privacy_client: PrivacyClientProtocol | None = None,
        user_profile_fetcher: UserProfileFetcherProtocol | None = None,
        episodic_searcher: EpisodicSearchProtocol | None = None,
    ) -> None:
        super().__init__(name=name)

        if privacy_client is None:
            from services.privacy_service import privacy_service as _ps
            privacy_client = _ps

        self._privacy_client: PrivacyClientProtocol = privacy_client
        self._user_profile_fetcher: UserProfileFetcherProtocol = (
            user_profile_fetcher if user_profile_fetcher is not None else get_user_profile
        )
        self._episodic_searcher: EpisodicSearchProtocol = (
            episodic_searcher if episodic_searcher is not None else search_similar_memories
        )

        self._circuit_breakers: dict[str, CircuitBreaker] = {
            "working_memory": CircuitBreaker(
                failure_threshold=_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                open_seconds=_CIRCUIT_BREAKER_OPEN_SECONDS,
            ),
            "user_profile": CircuitBreaker(
                failure_threshold=_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                open_seconds=_CIRCUIT_BREAKER_OPEN_SECONDS,
            ),
            "episodic_memories": CircuitBreaker(
                failure_threshold=_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                open_seconds=_CIRCUIT_BREAKER_OPEN_SECONDS,
            ),
            "episodic_memories_fallback": CircuitBreaker(
                failure_threshold=_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
                open_seconds=_CIRCUIT_BREAKER_OPEN_SECONDS,
            ),
        }

    # ── Main entry point ──────────────────────────────────────────────────────

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        session_state = ctx.session.state

        user_id = self._clean_text(session_state.get("user_id"))
        session_id = self._clean_text(session_state.get("session_id"))
        current_text = self._clean_text(session_state.get("current_transcript"))

        fallback_context = self._build_fallback_context()

        if not user_id:
            logger.warning(
                "MemoryAgent: missing user_id; skipping retrieval",
                extra={
                    "memory_source": "memory_agent",
                    "session_id": session_id,
                    "status": "skipped_missing_user_id",
                },
            )
            yield self._build_state_event(
                memory_context=fallback_context,
                memory_system_status="skipped_missing_user_id",
            )
            return

        # FIX: replaced asyncio.TaskGroup with asyncio.gather(return_exceptions=True).
        #
        # TaskGroup cancels ALL sibling tasks the moment any one of them raises —
        # the exact opposite of what we need here. If the profile fetch times out,
        # we still want working memory and episodic results to complete and return
        # partial context to the user.
        #
        # gather(return_exceptions=True) keeps every fetch fully independent.
        # DO NOT switch this back to TaskGroup.
        raw_results = await asyncio.gather(
            self._fetch_working_memory(session_id=session_id, user_id=user_id),
            self._fetch_user_profile(user_id=user_id, session_id=session_id),
            self._fetch_episodic_memories(
                user_id=user_id,
                query_text=current_text,
                session_id=session_id,
            ),
            return_exceptions=True,
        )

        # Each slot is either a (data, SourceResult) tuple or a BaseException.
        working_memory, working_meta = self._unpack_gather_slot(
            raw_results[0], label="working_memory", empty=[]
        )
        user_profile, profile_meta = self._unpack_gather_slot(
            raw_results[1], label="user_profile", empty=None
        )
        episodic_memories, episodic_meta = self._unpack_gather_slot(
            raw_results[2], label="episodic_memories", empty=[]
        )

        # Fallback episodic search: when no transcript is available to embed,
        # use the tail of working memory as the query instead.
        if not episodic_memories and not current_text and working_memory:
            fallback_query = self._build_fallback_query(working_memory)
            if fallback_query:
                fallback_raw = await self._fetch_episodic_memories(
                    user_id=user_id,
                    query_text=fallback_query,
                    session_id=session_id,
                    source_label="episodic_memories_fallback",
                    timeout_seconds=_FALLBACK_EPISODIC_TIMEOUT_SECONDS,
                )
                # _fetch_episodic_memories always returns a tuple because it has
                # its own internal retry wrapper — it never raises to the caller.
                if not isinstance(fallback_raw, BaseException):
                    episodic_fallback, episodic_fallback_meta = fallback_raw
                    if episodic_fallback:
                        episodic_memories = episodic_fallback
                    episodic_meta = self._merge_source_results(
                        primary=episodic_meta,
                        secondary=episodic_fallback_meta,
                    )

        outcome = MemoryFetchOutcome(
            working_memory=working_memory,
            working_meta=working_meta,
            user_profile=user_profile,
            profile_meta=profile_meta,
            episodic_memories=episodic_memories,
            episodic_meta=episodic_meta,
        )

        memory_context_string = self._assemble_memory_context(
            working_memory=outcome.working_memory,
            user_profile=outcome.user_profile,
            episodic_memories=outcome.episodic_memories,
            fallback_context=fallback_context,
        )

        memory_status = self._compute_memory_status(outcome)

        logger.info(
            "MemoryAgent assembled context",
            extra={
                "memory_source": "memory_agent",
                "session_id": session_id,
                "working_count": len(outcome.working_memory),
                "profile_present": bool(outcome.user_profile),
                "episodic_count": len(outcome.episodic_memories),
                "memory_system_status": memory_status,
                "working_ok": outcome.working_meta.ok,
                "profile_ok": outcome.profile_meta.ok,
                "episodic_ok": outcome.episodic_meta.ok,
            },
        )

        yield self._build_state_event(
            memory_context=memory_context_string,
            memory_system_status=memory_status,
        )

    # ── Event builder ─────────────────────────────────────────────────────────

    def _build_state_event(
        self,
        *,
        memory_context: str,
        memory_system_status: str,
    ) -> Event:
        # NOTE TO TEAM: this writes two session.state keys — memory_context
        # (original spec) and memory_system_status (new, added with this upgrade).
        # memory_system_status values: "ok" | "degraded_partial_failure" |
        # "degraded_all_sources_failed" | "skipped_missing_user_id".
        # The orchestrator and other agents can read this for observability but
        # must not depend on it for core routing logic.
        return Event(
            author=self.name,
            actions=EventActions(
                state_delta={
                    "memory_context": memory_context,
                    "memory_system_status": memory_system_status,
                }
            ),
        )

    # ── Fetch helpers ─────────────────────────────────────────────────────────

    async def _fetch_working_memory(
        self,
        *,
        session_id: str,
        user_id: str,
    ) -> tuple[list[WorkingMemoryEntry], SourceResult]:
        if not session_id:
            logger.warning(
                "MemoryAgent: missing session_id; working memory skipped",
                extra={"memory_source": "working_memory", "user_id": user_id},
            )
            return [], SourceResult(ok=True, item_count=0)

        async def operation() -> list[WorkingMemoryEntry]:
            rows = await self._privacy_client.get_recent_transcript(
                session_id=session_id,
                max_turns=_MAX_WORKING_TURNS,
            )
            if not isinstance(rows, list):
                raise TypeError("working memory source returned non-list")

            parsed: list[WorkingMemoryEntry] = []
            for row in rows:
                try:
                    raw = RawWorkingMemoryRow.model_validate(row)
                    parsed.append(WorkingMemoryEntry(speaker=raw.speaker, text=raw.text))
                except ValidationError:
                    logger.warning(
                        "MemoryAgent: invalid working memory row skipped",
                        extra={"memory_source": "working_memory", "session_id": session_id},
                    )
            return parsed

        return await self._run_source_with_retry(
            label="working_memory",
            session_id=session_id,
            user_id=user_id,
            timeout_seconds=_WORKING_MEMORY_TIMEOUT_SECONDS,
            operation=operation,
        )

    async def _fetch_user_profile(
        self,
        *,
        user_id: str,
        session_id: str,
    ) -> tuple[UserProfile | None, SourceResult]:
        async def operation() -> UserProfile | None:
            payload = await self._user_profile_fetcher(user_id)
            if payload is None:
                return None
            if not isinstance(payload, dict):
                raise TypeError("user profile source returned non-dict")
            return UserProfile.model_validate(payload)

        return await self._run_source_with_retry(
            label="user_profile",
            session_id=session_id,
            user_id=user_id,
            timeout_seconds=_USER_PROFILE_TIMEOUT_SECONDS,
            operation=operation,
        )

    async def _fetch_episodic_memories(
        self,
        *,
        user_id: str,
        query_text: str,
        session_id: str,
        source_label: str = "episodic_memories",
        timeout_seconds: float = _EPISODIC_MEMORY_TIMEOUT_SECONDS,
    ) -> tuple[list[EpisodicMemoryResult], SourceResult]:
        if not user_id or not query_text:
            return [], SourceResult(ok=True, item_count=0)

        async def operation() -> list[EpisodicMemoryResult]:
            rows = await self._episodic_searcher(
                user_id=user_id,
                query_text=query_text[:_MAX_QUERY_CHARS],
                top_k=_MAX_EPISODIC_RESULTS,
                min_similarity=_MIN_SIMILARITY,
            )
            if not isinstance(rows, list):
                raise TypeError("episodic search returned non-list")

            parsed_results: list[EpisodicMemoryResult] = []
            for row in rows:
                try:
                    raw = RawEpisodicRow.model_validate(row)
                except ValidationError:
                    logger.warning(
                        "MemoryAgent: invalid episodic row skipped",
                        extra={"memory_source": source_label, "session_id": session_id},
                    )
                    continue

                if raw.similarity < _MIN_SIMILARITY:
                    continue

                parsed_results.append(
                    EpisodicMemoryResult(
                        memory_id=raw.id,
                        emotion_label=raw.emotion_label,
                        similarity_score=raw.similarity,
                        created_at=raw.created_at,
                    )
                )
            return parsed_results

        return await self._run_source_with_retry(
            label=source_label,
            session_id=session_id,
            user_id=user_id,
            timeout_seconds=timeout_seconds,
            operation=operation,
        )

    # ── Retry + circuit breaker core ──────────────────────────────────────────

    async def _run_source_with_retry(
        self,
        *,
        label: str,
        session_id: str,
        user_id: str,
        timeout_seconds: float,
        operation: Callable[[], Awaitable[Any]],
    ) -> tuple[Any, SourceResult]:
        # FIX: use .get() so an unknown label never raises KeyError and bypasses
        # all retry/fallback logic. Proceeds unguarded with a logged error instead.
        breaker = self._circuit_breakers.get(label)
        if breaker is None:
            logger.error(
                "MemoryAgent: no circuit breaker registered for label %r; proceeding unguarded",
                label,
                extra={"memory_source": label, "session_id": session_id},
            )

        start = time.perf_counter()

        for attempt in range(1, _RETRY_ATTEMPTS + 1):
            if breaker is not None and not breaker.allow_request():
                snapshot = breaker.snapshot()
                logger.warning(
                    "Memory fetch short-circuited by open breaker",
                    extra={
                        "memory_source": label,
                        "session_id": session_id,
                        "breaker_state": "open",
                        "breaker_failures": snapshot["consecutive_failures"],
                        "open_remaining_seconds": snapshot["open_remaining_seconds"],
                    },
                )
                return self._empty_result_for_label(label), SourceResult(
                    ok=False,
                    timed_out=False,
                    retries_used=max(0, attempt - 1),
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    item_count=0,
                    error_type="CircuitOpen",
                )

            try:
                result = await asyncio.wait_for(operation(), timeout=timeout_seconds)
                latency_ms = (time.perf_counter() - start) * 1000.0
                item_count = len(result) if isinstance(result, list) else None

                if breaker is not None:
                    breaker.record_success()

                logger.info(
                    "Memory fetch success",
                    extra={
                        "memory_source": label,
                        "session_id": session_id,
                        "attempt": attempt,
                        "latency_ms": round(latency_ms, 2),
                        "item_count": item_count,
                    },
                )

                return result, SourceResult(
                    ok=True,
                    timed_out=False,
                    retries_used=attempt - 1,
                    latency_ms=latency_ms,
                    item_count=item_count,
                    error_type=None,
                )

            except asyncio.CancelledError:
                # Never swallow CancelledError — propagate immediately.
                raise

            except asyncio.TimeoutError:
                if breaker is not None:
                    breaker.record_failure()

                if attempt < _RETRY_ATTEMPTS:
                    await asyncio.sleep(self._compute_retry_delay(attempt))
                    continue

                latency_ms = (time.perf_counter() - start) * 1000.0
                logger.error(
                    "Memory fetch timeout (all retries exhausted)",
                    extra={
                        "memory_source": label,
                        "session_id": session_id,
                        "timeout_seconds": timeout_seconds,
                        "attempt": attempt,
                        "latency_ms": round(latency_ms, 2),
                    },
                )
                return self._empty_result_for_label(label), SourceResult(
                    ok=False,
                    timed_out=True,
                    retries_used=attempt - 1,
                    latency_ms=latency_ms,
                    item_count=0,
                    error_type="TimeoutError",
                )

            except Exception as exc:
                if breaker is not None:
                    breaker.record_failure()

                if attempt < _RETRY_ATTEMPTS:
                    await asyncio.sleep(self._compute_retry_delay(attempt))
                    continue

                latency_ms = (time.perf_counter() - start) * 1000.0
                logger.exception(
                    "Memory fetch failed (all retries exhausted)",
                    extra={
                        "memory_source": label,
                        "session_id": session_id,
                        "attempt": attempt,
                        "latency_ms": round(latency_ms, 2),
                        "error_type": type(exc).__name__,
                    },
                )
                return self._empty_result_for_label(label), SourceResult(
                    ok=False,
                    timed_out=False,
                    retries_used=attempt - 1,
                    latency_ms=latency_ms,
                    item_count=0,
                    error_type=type(exc).__name__,
                )

        raise RuntimeError(f"MemoryAgent: retry loop for '{label}' exited without returning")

    # ── Assembly ──────────────────────────────────────────────────────────────

    def _assemble_memory_context(
        self,
        *,
        working_memory: list[WorkingMemoryEntry],
        user_profile: UserProfile | None,
        episodic_memories: list[EpisodicMemoryResult],
        fallback_context: str,
    ) -> str:
        try:
            memory_ctx = MemoryContext(
                working_memory=working_memory,
                user_profile=user_profile,
                episodic_memories=episodic_memories,
            )
            prompt_string = memory_ctx.to_prompt_string()
            if isinstance(prompt_string, str) and prompt_string.strip():
                return prompt_string.strip()
        except Exception:
            logger.exception("MemoryAgent: MemoryContext assembly failed")

        return fallback_context

    def _build_fallback_context(self) -> str:
        try:
            fallback = MemoryContext().to_prompt_string()
            if isinstance(fallback, str) and fallback.strip():
                return fallback.strip()
        except Exception:
            logger.exception("MemoryAgent: fallback MemoryContext rendering failed")

        return "No memory context yet."

    def _compute_memory_status(self, outcome: MemoryFetchOutcome) -> str:
        ok_count = sum([
            outcome.working_meta.ok,
            outcome.profile_meta.ok,
            outcome.episodic_meta.ok,
        ])
        if ok_count == 3:
            return "ok"
        if ok_count == 0:
            return "degraded_all_sources_failed"
        return "degraded_partial_failure"

    def _merge_source_results(
        self,
        *,
        primary: SourceResult,
        secondary: SourceResult,
    ) -> SourceResult:
        return SourceResult(
            ok=primary.ok or secondary.ok,
            timed_out=primary.timed_out and secondary.timed_out,
            retries_used=max(primary.retries_used, secondary.retries_used),
            latency_ms=max(primary.latency_ms, secondary.latency_ms),
            item_count=max(primary.item_count or 0, secondary.item_count or 0),
            error_type=secondary.error_type if not secondary.ok else primary.error_type,
        )

    def _build_fallback_query(self, working_memory: list[WorkingMemoryEntry]) -> str:
        parts = [
            entry.text
            for entry in working_memory[-_FALLBACK_QUERY_TURNS:]
            if isinstance(entry.text, str) and entry.text.strip()
        ]
        return _build_sentence_aware_query(parts, _MAX_QUERY_CHARS)

    def _unpack_gather_slot(
        self,
        raw: Any,
        *,
        label: str,
        empty: Any,
    ) -> tuple[Any, SourceResult]:
        """Unpack one slot from asyncio.gather(return_exceptions=True).
        Returns either the coroutine's (data, SourceResult) tuple, or
        the safe empty fallback if the slot holds a BaseException.
        """
        if isinstance(raw, BaseException):
            logger.error(
                "MemoryAgent: %s raised an unhandled exception outside retry wrapper: %s",
                label,
                raw,
                extra={"memory_source": label},
            )
            return empty, SourceResult(ok=False, item_count=0, error_type=type(raw).__name__)
        return raw

    @staticmethod
    def _clean_text(value: object) -> str:
        if not isinstance(value, str):
            return ""
        return _normalize_text(value)

    @staticmethod
    def _compute_retry_delay(attempt: int) -> float:
        """Exponential backoff with ±25% jitter to prevent thundering herd on Supabase."""
        exp_delay = min(
            _RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
            _RETRY_MAX_DELAY_SECONDS,
        )
        jitter = random.uniform(0.0, exp_delay * 0.25)
        return exp_delay + jitter

    @staticmethod
    def _empty_result_for_label(label: str) -> Any:
        if label == "user_profile":
            return None
        return []


# ── Singleton ─────────────────────────────────────────────────────────────────
memory_agent = MemoryAgent(name="memory_agent")
