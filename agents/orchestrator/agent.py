"""BEAN AI v5 — Orchestrator Agent.

The central coordinator that runs every conversation turn:

  Phase 1: Safety pre-screen (fast keyword check — no LLM)
  Phase 2: Route decision (Gemini Flash — cheap)
  Phase 3: Memory context fetch (Supabase profile + episodic vector search)
  Phase 4: Response generation — dispatches to the correct sub-agent:
             casual   → CasualChatAgent   (Flash, 2-sentence persona)
             therapy  → TherapyAgent      (Pro + RAG CBT/DBT techniques)
             task     → TaskAgent         (Flash + Google Calendar tools)
             music    → MusicAgent        (Flash + ESP32 music commands)
             alert    → crisis response   (immediate, no sub-agent needed)
  Phase 5: Post-response tasks (non-blocking, asyncio.gather):
             • Safety assessment (Pro) — skipped for alert route (already done)
             • Memory write (Flash fact extraction + embedding)
             • Transcript store (temporary, 24h TTL)
             • Session metadata update

Design principles:
  - Never delays user response for safety/memory background tasks
  - Pro LLM only for therapy/alert routes
  - All background tasks run concurrently via asyncio.gather
  - Any background task failure is logged but never surfaces to user
  - Sub-agents receive memory_context + current_emotion via session_state

Bugs fixed vs. original:
  - check_crisis_keywords was imported lazily inside _run_async_impl on every
    call → moved to module-level import.
  - Return type annotation was `adk_types.AsyncGenerator` — adk_types is
    google.genai.types, not typing; AsyncGenerator doesn't exist there.
    Fixed to AsyncGenerator[Event, None] from collections.abc.
  - Orchestrator yielded Content objects directly.  The ADK BaseAgent
    framework and the WebSocket handler both expect Event objects (with
    event.content set).  Yielding Content causes AttributeError: 'Content'
    object has no attribute 'content' in the handler.  Fixed to yield Event.
  - Double Pro-LLM safety call on crisis: original code called both
    _handle_crisis (safety_service.assess_turn) AND _run_background_tasks
    (which also calls _background_safety_check → safety_service.assess_turn).
    This doubled guardian SMS risk and wasted Pro LLM quota.  Fixed: skip
    _background_safety_check for the alert route.
  - Missing top-level imports: Event from google.adk.events.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from services.llm_service import generate_json
from services.privacy_service import privacy_service
from services.safety_service import (
    check_crisis_keywords,  # FIX: was a deferred import inside _run_async_impl
    safety_service,
)
from services.supabase_client import get_service_client, get_user_profile
from shared.exceptions import CrisisDetectedError
from shared.schemas import UserProfile as UserProfileSchema

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CRISIS_RESPONSE = (
    "I can hear that you're going through something really hard right now, "
    "and I'm glad you're talking to me. Please know you don't have to face "
    "this alone. If you're in immediate danger, please call emergency services "
    "or a crisis helpline. Your support person has been notified and cares "
    "about you very much. I'm here. 💙"
)

ROUTING_SYSTEM = """You are a routing classifier for BEAN, a mental health companion robot.
Classify the user's message into exactly one route.

Routes:
  casual  — general conversation, small talk, sharing how their day was
  therapy — the user is expressing sadness, distress, anxiety, loneliness, or emotional pain
  task    — setting reminders, managing to-dos, scheduling
  music   — requesting music, asking to play/stop/change songs
  alert   — explicit self-harm language, crisis, suicidal ideation

Rules:
- When in doubt between casual and therapy, choose therapy
- Only choose alert if there is clear, explicit risk language
- Consider the emotion when deciding between casual and therapy

Respond ONLY with valid JSON — no markdown, no explanation:
{"route": "casual|therapy|task|music|alert", "confidence": 0.0-1.0}"""

ROUTING_PROMPT = """\
User message: {user_text}
Detected emotion: {emotion}
Turn number: {turn_number}
Recent route distribution: {route_distribution}

Classify this message."""

_VALID_ROUTES: frozenset[str] = frozenset(
    {"casual", "therapy", "task", "music", "alert"}
)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# ADK runner helper
# ─────────────────────────────────────────────────────────────────────────────


async def _run_sub_agent(
    agent: BaseAgent,
    user_id: str,
    user_text: str,
    state: dict,
) -> tuple[str, dict]:
    """Run a sub-agent via InMemoryRunner and return (response_text, final_state).

    ADK sub-agents (BaseAgent subclasses) must be invoked through a Runner —
    calling agent.run_async() directly is not valid.  InMemoryRunner gives us
    a lightweight in-process session store that is discarded after each turn.

    State is seeded at session-creation time so ctx.session.state is populated
    before _run_async_impl is called.  We retrieve the updated session after the
    run to read any state mutations the agent made (e.g. music_command).
    """
    runner = InMemoryRunner(agent=agent)
    session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=user_id,
        state=state,
    )

    response_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_text)],
        ),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    response_text = part.text

    # Read back any state mutations the agent made during its run.
    updated = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session.id,
    )
    final_state: dict = dict(updated.state) if updated else state
    return response_text, final_state


class BEANOrchestrator(BaseAgent):
    """Main BEAN AI orchestrator — coordinates all agents per conversation turn."""

    # ── Main entry point ──────────────────────────────────────────────────────

    async def _run_async_impl(  # type: ignore[override]
        self, ctx: InvocationContext
    ) -> AsyncGenerator[
        Event, None
    ]:  # FIX: was adk_types.AsyncGenerator (wrong module)
        state = ctx.session.state
        user_id: str = state["user_id"]
        session_id: str = state["session_id"]
        user_text: str = state.get("current_transcript", "")
        emotion: str = state.get("current_emotion", "neutral")
        turn_number: int = state.get("turn_count", 0) + 1

        if not user_text.strip():
            yield Event(author=self.name, content=None)
            return

        turn_id = str(uuid.uuid4())
        state["turn_id"] = turn_id
        state["turn_count"] = turn_number

        # ── Phase 1: Fast crisis keyword pre-screen ───────────────────────────
        # check_crisis_keywords is now a module-level import (was deferred).
        has_crisis, _ = check_crisis_keywords(user_text)
        if has_crisis:
            state["route"] = "alert"
            state["response_text"] = CRISIS_RESPONSE

            # FIX: _handle_crisis calls safety_service.assess_turn (Pro LLM).
            # We must NOT also call _background_safety_check (same Pro LLM call)
            # — that would double-fire and risk sending two guardian SMSes.
            # Pass route="alert" to _run_background_tasks, which now skips
            # _background_safety_check for the alert route.
            asyncio.create_task(
                self._handle_crisis(user_id, session_id, user_text, emotion),
                name=f"crisis-{turn_id[:8]}",
            )

            yield Event(  # FIX: yield Event, not Content directly
                author=self.name,
                content=genai_types.Content(
                    parts=[genai_types.Part(text=CRISIS_RESPONSE)]
                ),
            )

            asyncio.create_task(
                self._run_background_tasks(
                    user_id=user_id,
                    session_id=session_id,
                    user_text=user_text,
                    response_text=CRISIS_RESPONSE,
                    emotion=emotion,
                    route="alert",  # skips safety check in background tasks
                    turn_number=turn_number,
                ),
                name=f"bg-alert-{turn_id[:8]}",
            )
            return

        # ── Phase 2: Route decision (Gemini Flash) ────────────────────────────
        route = await self._decide_route(user_text, emotion, turn_number, state)
        state["route"] = route

        # ── Phase 3: Memory context ───────────────────────────────────────────
        memory_context = await self._fetch_memory_context(user_id, user_text)
        state["memory_context"] = memory_context

        # ── Phase 4: Dispatch to sub-agent ────────────────────────────────────
        response_text, music_command = await self._dispatch(
            route=route,
            state=state,
            user_id=user_id,
            session_id=session_id,
            user_text=user_text,
            emotion=emotion,
            memory_context=memory_context,
        )

        state["response_text"] = response_text
        state["route_distribution"] = {
            **state.get("route_distribution", {}),
            route: state.get("route_distribution", {}).get(route, 0) + 1,
        }
        if music_command is not None:
            state["music_command"] = music_command

        yield Event(  # FIX: yield Event with content, not bare Content
            author=self.name,
            content=genai_types.Content(parts=[genai_types.Part(text=response_text)]),
        )

        # ── Phase 5: Non-blocking background tasks ────────────────────────────
        asyncio.create_task(
            self._run_background_tasks(
                user_id=user_id,
                session_id=session_id,
                user_text=user_text,
                response_text=response_text,
                emotion=emotion,
                route=route,
                turn_number=turn_number,
            ),
            name=f"bg-{route}-{turn_id[:8]}",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Route decision
    # ─────────────────────────────────────────────────────────────────────────

    async def _decide_route(
        self,
        user_text: str,
        emotion: str,
        turn_number: int,
        state: dict,
    ) -> str:
        """Classify route using Gemini Flash (cheap tier, inline call).

        Inline rather than sub-agent to avoid ADK session-state isolation issues
        when reading context back from a child session.
        """
        prompt = ROUTING_PROMPT.format(
            user_text=user_text[:300],
            emotion=emotion,
            turn_number=turn_number,
            route_distribution=state.get("route_distribution", {}),
        )
        try:
            result = await generate_json(
                task="routing",
                prompt=prompt,
                system=ROUTING_SYSTEM,
            )
            route = result.get("route", "casual")
            if route not in _VALID_ROUTES:
                logger.warning(
                    "Routing returned invalid route '%s', defaulting to casual", route
                )
                route = "casual"
            confidence = float(result.get("confidence", 0.8))
            state["routing_confidence"] = confidence
            logger.debug(
                "Route: %s (confidence=%.2f) turn=%d", route, confidence, turn_number
            )
            return route
        except Exception as exc:
            logger.warning("Routing failed, defaulting to casual: %s", exc)
            return "casual"

    # ─────────────────────────────────────────────────────────────────────────
    # Sub-agent dispatch
    # ─────────────────────────────────────────────────────────────────────────

    async def _dispatch(
        self,
        route: str,
        state: dict,
        user_id: str,
        session_id: str,
        user_text: str,
        emotion: str,
        memory_context: str,
    ) -> tuple[str, dict | None]:
        """Dispatch to the correct sub-agent. Returns (response_text, music_command).

        Sub-agents are BaseAgent subclasses and must be invoked via InMemoryRunner —
        calling agent.run_async() directly is invalid (wrong signature).  We create
        a fresh ephemeral InMemoryRunner per turn, seed it with the sub-state dict,
        and read response_text / music_command back from the final session state.
        """
        _exclude_keys = frozenset(
            {"current_transcript", "current_emotion", "memory_context", "response_text"}
        )
        sub_state: dict = {
            "user_id": user_id,
            "session_id": session_id,
            "current_transcript": user_text,
            "current_emotion": emotion,
            "memory_context": memory_context,
            "response_text": "",
            **{k: v for k, v in state.items() if k not in _exclude_keys},
        }

        response_text = ""
        music_command: dict | None = None
        final_state: dict = sub_state

        try:
            if route == "casual":
                from agents.casual_chat.agent import casual_chat_agent

                response_text, final_state = await _run_sub_agent(
                    casual_chat_agent, user_id, user_text, sub_state
                )

            elif route == "therapy":
                from agents.therapeutic_convo.agent import therapy_agent

                response_text, final_state = await _run_sub_agent(
                    therapy_agent, user_id, user_text, sub_state
                )

            elif route == "task":
                from agents.task.agent import task_agent

                sub_state["calendar_access_token"] = await self._get_calendar_token(
                    user_id
                )
                response_text, final_state = await _run_sub_agent(
                    task_agent, user_id, user_text, sub_state
                )

            elif route == "music":
                from agents.music.agent import music_agent

                response_text, final_state = await _run_sub_agent(
                    music_agent, user_id, user_text, sub_state
                )
                music_command = final_state.get("music_command")

        except Exception as exc:
            logger.error("Sub-agent dispatch failed [route=%s]: %s", route, exc)

        response_text = response_text or final_state.get("response_text", "")

        if not response_text:
            response_text = await self._fallback_response(
                route, user_text, emotion, memory_context
            )

        return response_text, music_command

    async def _fallback_response(
        self,
        route: str,
        user_text: str,
        emotion: str,
        memory_context: str,
    ) -> str:
        """Direct LLM fallback when a sub-agent fails to produce output."""
        from services.llm_service import generate

        _FALLBACKS: dict[str, tuple[str, str]] = {
            "casual": (
                "casual_chat",
                "You are BEAN, a warm AI companion for teenagers. Keep it to 2 sentences max.",
            ),
            "therapy": (
                "therapeutic_chat",
                "You are BEAN, a supportive companion. Validate their feelings in 3 sentences max.",
            ),
            "task": (
                "task_management",
                "You are BEAN, helping with tasks. Confirm what you'll help with in 2 sentences.",
            ),
            "music": (
                "music_selection",
                "You are BEAN. Tell them what music you're putting on in 1-2 sentences.",
            ),
        }
        task, system = _FALLBACKS.get(
            route,
            ("casual_chat", "You are BEAN, a warm AI companion."),
        )
        prompt = f"Memory:\n{memory_context}\n\nEmotion: {emotion}\n\nUser: {user_text}\nBEAN:"
        try:
            return await generate(task=task, prompt=prompt, system=system)
        except Exception as exc:
            logger.error("Fallback response failed: %s", exc)
            return (
                "I'm here with you — could you say that again? "
                "I want to make sure I'm really listening."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # Memory context
    # ─────────────────────────────────────────────────────────────────────────

    async def _fetch_memory_context(self, user_id: str, user_text: str) -> str:
        """Fetch user profile + relevant episodic memories for prompt injection."""
        from services.embedding_service import search_similar_memories

        profile_str = ""
        episodic_str = ""

        try:
            profile = await get_user_profile(user_id)
            if profile:
                profile_obj = UserProfileSchema(**profile)
                profile_str = profile_obj.to_context_string()
        except Exception as exc:
            logger.debug("Profile fetch failed (non-critical): %s", exc)

        try:
            memories = await search_similar_memories(
                user_id=user_id,
                query_text=user_text[:300],
                top_k=3,
                min_similarity=0.72,
            )
            if memories:
                episodic_str = f"Relevant context: {len(memories)} similar past interactions found."
        except Exception as exc:
            logger.debug("Episodic memory search failed (non-critical): %s", exc)

        parts = []
        if profile_str and profile_str != "No profile yet.":
            parts.append(f"User profile:\n{profile_str}")
        if episodic_str:
            parts.append(episodic_str)

        return "\n\n".join(parts) if parts else ""

    async def _get_calendar_token(self, user_id: str) -> str | None:
        """Safely fetch the user's Google Calendar OAuth token."""
        try:
            from services.calendar_service import get_calendar_token

            return await get_calendar_token(user_id)
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────────
    # Crisis handling
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_crisis(
        self, user_id: str, session_id: str, user_text: str, emotion: str
    ) -> None:
        """Trigger full safety assessment with vulnerability flag raised.

        This is the ONLY place safety_service.assess_turn is called for the
        alert route.  _background_safety_check is skipped for alert turns to
        prevent a duplicate Pro-LLM call and duplicate guardian SMS dispatch.
        """
        try:
            await safety_service.assess_turn(
                user_id=user_id,
                session_id=session_id,
                user_text=user_text,
                emotion=emotion,
                emotion_trend=[emotion],
                turn_number=1,
                vulnerability_flag=True,
            )
        except CrisisDetectedError:
            # Expected — crisis was already handled upstream.
            pass
        except Exception as exc:
            logger.error("Crisis handling failed: %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Background tasks
    # ─────────────────────────────────────────────────────────────────────────

    async def _run_background_tasks(
        self,
        user_id: str,
        session_id: str,
        user_text: str,
        response_text: str,
        emotion: str,
        route: str,
        turn_number: int,
    ) -> None:
        """Run all post-turn background tasks concurrently.

        FIX: For route='alert', _background_safety_check is omitted.
             _handle_crisis already called safety_service.assess_turn for this
             turn, so running it again would:
               (a) waste a Gemini Pro API call, and
               (b) risk sending a second guardian SMS for the same event.
        """
        tasks = [
            self._background_memory_write(
                user_id, session_id, user_text, response_text, emotion
            ),
            self._background_store_transcript(
                session_id, user_id, user_text, response_text
            ),
            self._background_update_session(session_id, turn_number, emotion),
        ]

        if route != "alert":
            # Safety assessment already done inline for alert route.
            tasks.insert(
                0,
                self._background_safety_check(
                    user_id, session_id, user_text, emotion, turn_number
                ),
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    "Background task %d failed [route=%s]: %s", i, route, result
                )

    async def _background_safety_check(
        self,
        user_id: str,
        session_id: str,
        user_text: str,
        emotion: str,
        turn_number: int,
    ) -> None:
        """Standard safety check for non-alert conversation turns (Gemini Pro)."""
        try:
            await safety_service.assess_turn(
                user_id=user_id,
                session_id=session_id,
                user_text=user_text,
                emotion=emotion,
                emotion_trend=[emotion],
                turn_number=turn_number,
            )
        except CrisisDetectedError:
            logger.warning(
                "Crisis detected in background safety assessment [user=%s]",
                user_id[:8],
            )
        except Exception as exc:
            logger.error("Background safety check failed: %s", exc)

    async def _background_memory_write(
        self,
        user_id: str,
        session_id: str,
        user_text: str,
        response_text: str,
        emotion: str,
    ) -> None:
        """Extract facts and write episodic embedding (privacy-safe)."""
        from agents.memory_writer.agent import memory_writer_agent as mwa

        try:
            async for _ in mwa.run_async(
                user_id=user_id,
                session_id=session_id,
                session_state={
                    "user_id": user_id,
                    "session_id": session_id,
                    "current_transcript": user_text,
                    "response_text": response_text,
                    "current_emotion": emotion,
                },
            ):
                pass
        except Exception as exc:
            logger.error("Background memory write failed: %s", exc)

    async def _background_store_transcript(
        self,
        session_id: str,
        user_id: str,
        user_text: str,
        response_text: str,
    ) -> None:
        """Store both sides of the turn in the 24h temporary transcript."""
        try:
            await privacy_service.store_transcript_turn(
                session_id=session_id,
                user_id=user_id,
                speaker="user",
                text=user_text,
            )
            await privacy_service.store_transcript_turn(
                session_id=session_id,
                user_id=user_id,
                speaker="assistant",
                text=response_text,
            )
        except Exception as exc:
            logger.error("Transcript store failed: %s", exc)

    async def _background_update_session(
        self, session_id: str, turn_number: int, emotion: str
    ) -> None:
        """Update live session metrics in Supabase."""
        try:
            client = await get_service_client()
            await (
                client.table("sessions")
                .update({"turn_count": turn_number, "dominant_emotion": emotion})
                .eq("id", session_id)
                .execute()
            )
        except Exception as exc:
            logger.error("Session update failed: %s", exc)


# ── Singleton ─────────────────────────────────────────────────────────────────

orchestrator = BEANOrchestrator(name="bean_orchestrator")
