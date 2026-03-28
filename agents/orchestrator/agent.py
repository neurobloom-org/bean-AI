"""BEAN AI v1 — Orchestrator Agent.

The central coordinator that runs every conversation turn:

  Phase 1: Safety pre-screen (fast keyword check — no LLM)
  Phase 2: Route decision (routing_agent — Gemini Flash with retry + alert floor)
  Phase 3: Memory context fetch (Supabase profile + episodic vector search)
  Phase 4: Response generation — dispatches to the correct sub-agent:
             casual   → CasualChatAgent   (Flash, 2-sentence persona)
             therapy  → TherapyAgent      (Pro + RAG CBT/DBT techniques)
             task     → TaskAgent         (Flash + Supabase tasks table)
             music    → MusicAgent        (Flash + ESP32 music commands)
             alert    → crisis response   (immediate, no sub-agent needed)
  Phase 5: Post-response tasks (non-blocking, asyncio.gather):
             • Safety assessment (Pro) — skipped for alert route (already done)
             • Memory write (Flash fact extraction + embedding)
             • Transcript store (temporary, 24h TTL)
             • Session metadata update
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

from agents.routing.agent import routing_agent
from services.llm_service import generate
from services.privacy_service import privacy_service
from services.safety_service import (
    check_crisis_keywords,
    check_explicit_statement,
    get_post_alert_message,
    safety_service,
)
from services.supabase_client import get_service_client, get_user_profile
from shared.exceptions import CrisisDetectedError
from shared.schemas import UserProfile as UserProfileSchema

logger = logging.getLogger(__name__)

_ORCHESTRATOR_TASKS: set[asyncio.Task] = set()


def _create_tracked_task(coro, *, name: str) -> asyncio.Task:
    task = asyncio.create_task(coro, name=name)
    _ORCHESTRATOR_TASKS.add(task)
    task.add_done_callback(_ORCHESTRATOR_TASKS.discard)
    return task


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CRISIS_RESPONSE = (
    "I can hear that you're going through something really hard right now, "
    "and I'm glad you're talking to me. Please know you don't have to face "
    "this alone. If you're in immediate danger, please call emergency services "
    "or a crisis helpline. Your support person has been notified and cares "
    "about you very much. I'm here. "
)


# ─────────────────────────────────────────────────────────────────────────────
# ADK runner helper
# ─────────────────────────────────────────────────────────────────────────────


async def _run_sub_agent(
    agent: BaseAgent,
    user_id: str,
    user_text: str,
    state: dict,
) -> tuple[str, dict]:
    """Run a sub-agent via InMemoryRunner and return (response_text, final_state)."""
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

    updated = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id=user_id,
        session_id=session.id,
    )
    final_state: dict = dict(updated.state) if updated else state
    return response_text, final_state


class BEANOrchestrator(BaseAgent):
    """Main BEAN AI orchestrator — coordinates all agents per conversation turn."""

    async def _run_async_impl(  # type: ignore[override]
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
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

        # ── Task confirmation bypass ──────────────────────────────────────────
        if state.get("task_pending_confirmation"):
            from agents.task.agent import task_agent

            memory_context = await self._fetch_memory_context(user_id, user_text)
            state["memory_context"] = memory_context

            response_text, final_state = await _run_sub_agent(
                task_agent,
                user_id=user_id,
                user_text=user_text,
                state={
                    "user_id": user_id,
                    "session_id": session_id,
                    "current_transcript": user_text,
                    "current_emotion": emotion,
                    "memory_context": memory_context,
                    "response_text": "",
                    **{
                        k: v
                        for k, v in state.items()
                        if k
                        not in {
                            "current_transcript",
                            "current_emotion",
                            "memory_context",
                            "response_text",
                        }
                    },
                },
            )

            state.update(final_state)
            state["task_pending_confirmation"] = bool(
                final_state.get("task_pending_confirmation", False)
            )
            state["route"] = "task"
            state["response_text"] = response_text or final_state.get("response_text", "")
            state["route_distribution"] = {
                **state.get("route_distribution", {}),
                "task": state.get("route_distribution", {}).get("task", 0) + 1,
            }

            yield Event(
                author=self.name,
                content=genai_types.Content(
                    parts=[genai_types.Part(text=state["response_text"])]
                ),
            )

            _create_tracked_task(
                self._run_background_tasks(
                    user_id=user_id,
                    session_id=session_id,
                    user_text=user_text,
                    response_text=state["response_text"],
                    emotion=emotion,
                    route="task",
                    turn_number=turn_number,
                    is_minor=state.get("is_minor", False),
                ),
                name=f"bg-task-confirm-{turn_id[:8]}",
            )
            return

        # ── Reminder injection ────────────────────────────────────────────────
        pending_reminder = getattr(ctx.session, "pending_reminder", None) or state.get(
            "pending_reminder"
        )
        if pending_reminder:
            state["reminder_hint"] = (
                f"[Reminder hint — slip naturally into reply: "
                f"'{pending_reminder.get('title')}' is due soon]"
            )
            state["pending_reminder"] = None
            state["route"] = "casual"

        # ── Two-stage safety gate ─────────────────────────────────────────────
        transcript = str(state.get("current_transcript", ""))
        has_crisis_kw, _ = check_crisis_keywords(
            transcript, state.get("is_minor", False)
        )
        has_explicit, _ = check_explicit_statement(transcript)

        if has_crisis_kw or has_explicit:
            state["route"] = "alert"
            state["response_text"] = CRISIS_RESPONSE
            state["post_alert_message"] = get_post_alert_message()

            _create_tracked_task(
                self._handle_crisis(user_id, session_id, user_text, emotion),
                name=f"crisis-{turn_id[:8]}",
            )

            yield Event(
                author=self.name,
                content=genai_types.Content(
                    parts=[genai_types.Part(text=CRISIS_RESPONSE)]
                ),
            )

            _create_tracked_task(
                self._run_background_tasks(
                    user_id=user_id,
                    session_id=session_id,
                    user_text=user_text,
                    response_text=CRISIS_RESPONSE,
                    emotion=emotion,
                    route="alert",
                    turn_number=turn_number,
                    is_minor=state.get("is_minor", False),
                ),
                name=f"bg-alert-{turn_id[:8]}",
            )
            return

        # ── Phase 2: Route decision (routing_agent) ───────────────────────────
        _, routing_state = await _run_sub_agent(
            routing_agent,
            user_id=user_id,
            user_text=user_text,
            state={
                "user_id": user_id,
                "session_id": session_id,
                "current_transcript": user_text,
                "current_emotion": emotion,
                "turn_count": turn_number,
                "route_distribution": state.get("route_distribution", {}),
                "reminder_hint": state.get("reminder_hint"),
                "task_pending_confirmation": state.get("task_pending_confirmation"),
            },
        )
        route = routing_state.get("route", "casual")
        logger.info("DEBUG: reminder_hint=%s route=%s", state.get("reminder_hint"), route)
        if not state.get("reminder_hint"):
            state["route"] = route
        state["routing_confidence"] = routing_state.get("routing_confidence", 0.5)
        state["routing_used_fallback"] = routing_state.get("routing_used_fallback", False)

        # ── Phase 3: Memory context ───────────────────────────────────────────
        memory_context = await self._fetch_memory_context(user_id, user_text)
        state["memory_context"] = memory_context

        # ── Phase 4: Dispatch to sub-agent ────────────────────────────────────
        response_text = await self._dispatch(
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

        yield Event(
            author=self.name,
            content=genai_types.Content(parts=[genai_types.Part(text=response_text)]),
        )

        # ── Phase 5: Non-blocking background tasks ────────────────────────────
        _create_tracked_task(
            self._run_background_tasks(
                user_id=user_id,
                session_id=session_id,
                user_text=user_text,
                response_text=response_text,
                emotion=emotion,
                route=route,
                turn_number=turn_number,
                is_minor=state.get("is_minor", False),
            ),
            name=f"bg-{route}-{turn_id[:8]}",
        )

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
    ) -> str:
        _exclude_keys = frozenset(
            {"current_transcript", "current_emotion", "memory_context", "response_text"}
        )

        from shared.conversation_cache import get_history

        try:
            conversation_history = await get_history(session_id)
        except Exception:
            conversation_history = []

        sub_state: dict = {
            "user_id": user_id,
            "session_id": session_id,
            "current_transcript": user_text,
            "current_emotion": emotion,
            "memory_context": memory_context,
            "conversation_history": conversation_history,
            "response_text": "",
            **{k: v for k, v in state.items() if k not in _exclude_keys},
        }

        response_text = ""
        final_state: dict = sub_state

        try:
            if route == "casual":
                from agents.casual_chat.agent import reply_with_history
                try:
                    response_text = await reply_with_history(
                        user_text=user_text,
                        emotion=emotion,
                        conversation_history=conversation_history,
                        memory_context=memory_context,
                        reminder_hint=state.get("reminder_hint"),
                    )
                    state["reminder_hint"] = None
                    final_state = sub_state
                except Exception as exc:
                    logger.error("reply_with_history failed, falling back to agent: %s", exc)
                    from agents.casual_chat.agent import casual_chat_agent
                    response_text, final_state = await _run_sub_agent(
                        casual_chat_agent, user_id, user_text, sub_state
                    )

            elif route == "therapy":
                try:
                    from agents.active_listen.agent import active_listen_agent
                    await _run_sub_agent(
                        active_listen_agent, user_id, user_text, sub_state
                    )
                except Exception as exc:
                    logger.debug("Active listen skipped (non-critical): %s", exc)

                from agents.therapeutic_convo.agent import therapy_agent
                response_text, final_state = await _run_sub_agent(
                    therapy_agent, user_id, user_text, sub_state
                )

            elif route == "task":
                from agents.task.agent import task_agent
                response_text, final_state = await _run_sub_agent(
                    task_agent, user_id, user_text, sub_state
                )

            elif route == "music":
                from agents.music.agent import music_agent
                response_text, final_state = await _run_sub_agent(
                    music_agent, user_id, user_text, sub_state
                )
                # Forward music_action / music_mood / music_volume from the
                # sub-agent's state delta to the orchestrator's session state
                # so the WebSocket handler's _handle_music_action can read them.
                for key in ("music_action", "music_mood", "music_volume"):
                    if final_state.get(key) is not None:
                        state[key] = final_state[key]

        except Exception as exc:
            logger.error("Sub-agent dispatch failed [route=%s]: %s", route, exc)

        response_text = response_text or final_state.get("response_text", "")

        if not response_text:
            response_text = await self._fallback_response(
                route, user_text, emotion, memory_context
            )

        return response_text

    async def _fallback_response(
        self,
        route: str,
        user_text: str,
        emotion: str,
        memory_context: str,
    ) -> str:
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
        """Calendar OAuth not implemented — returns None (best-effort)."""
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Crisis handling
    # ─────────────────────────────────────────────────────────────────────────

    async def _handle_crisis(
        self, user_id: str, session_id: str, user_text: str, emotion: str
    ) -> None:
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
        is_minor: bool,
    ) -> None:
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
            tasks.insert(
                0,
                self._background_safety_check(
                    user_id, session_id, user_text, emotion, turn_number, is_minor
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
        is_minor: bool,
    ) -> None:
        try:
            has_crisis_kw, _ = check_crisis_keywords(user_text, is_minor)
            has_explicit, _ = check_explicit_statement(user_text)
            if not (has_crisis_kw or has_explicit):
                return

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
        from agents.memory_writer.agent import memory_writer_agent

        try:
            await _run_sub_agent(
                memory_writer_agent,
                user_id=user_id,
                user_text=user_text,
                state={
                    "user_id": user_id,
                    "session_id": session_id,
                    "current_transcript": user_text,
                    "response_text": response_text,
                    "current_emotion": emotion,
                },
            )
        except Exception as exc:
            logger.error("Background memory write failed: %s", exc)

    async def _background_store_transcript(
        self,
        session_id: str,
        user_id: str,
        user_text: str,
        response_text: str,
    ) -> None:
        from shared.conversation_cache import append_turn

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
            await append_turn(session_id, "user", user_text)
            await append_turn(session_id, "bean", response_text)
        except Exception as exc:
            logger.error("Transcript store failed: %s", exc)

    async def _background_update_session(
        self, session_id: str, turn_number: int, emotion: str
    ) -> None:
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