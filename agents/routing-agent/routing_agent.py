import logging
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types as adk_types

from services.llm_service import generate_json

logger = logging.getLogger(__name__)

VALID_ROUTES = {"casual", "therapy", "task", "music", "alert"}

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
- Use the provided emotional context and recent route history
- Respond ONLY with valid JSON

Return:
{"route": "casual|therapy|task|music|alert", "confidence": 0.0}
"""

ROUTING_PROMPT = """User message: {user_text}
Detected emotion: {emotion}
Turn number: {turn_number}
Recent route distribution: {route_distribution}
Recent emotion trend: {emotion_trend}

Classify this message."""
    

def _safe_confidence(value: Any, default: float = 0.8) -> float:
    try:
        conf = float(value)
        return max(0.0, min(1.0, conf))
    except (TypeError, ValueError):
        return default


def _fallback_route(emotion: str) -> str:
    emotion = (emotion or "").strip().lower()
    if emotion in {"sad", "anxious", "angry", "distressed", "lonely", "overwhelmed"}:
        return "therapy"
    return "casual"


class RoutingAgent(BaseAgent):
    """Lightweight routing classifier using Gemini Flash."""

    async def _run_async_impl(self, ctx: InvocationContext) -> adk_types.AsyncGenerator:
        state = ctx.session.state

        user_text = str(state.get("current_transcript", "")).strip()
        emotion = str(state.get("current_emotion", "neutral")).strip().lower()
        turn_number = int(state.get("turn_count", 0))
        route_distribution = state.get("route_distribution", {})
        emotion_trend = state.get("recent_emotions", [])

        if not user_text:
            state["route"] = "casual"
            state["routing_confidence"] = 1.0
            yield adk_types.Content(parts=[])
            return

        prompt = ROUTING_PROMPT.format(
            user_text=user_text[:300],
            emotion=emotion,
            turn_number=turn_number,
            route_distribution=route_distribution,
            emotion_trend=emotion_trend[-5:] if isinstance(emotion_trend, list) else [],
        )

        try:
            result = await generate_json(
                task="routing",
                prompt=prompt,
                system=ROUTING_SYSTEM,
            )

            if not isinstance(result, dict):
                raise ValueError("generate_json returned non-dict result")

            route = str(result.get("route", "casual")).strip().lower()
            confidence = _safe_confidence(result.get("confidence", 0.8))

            if route not in VALID_ROUTES:
                logger.warning("Invalid route '%s' returned; using safer fallback", route)
                route = _fallback_route(emotion)

            if confidence < 0.45 and route == "casual":
                route = _fallback_route(emotion)

            state["route"] = route
            state["routing_confidence"] = confidence
            logger.debug("Route decision: %s (confidence=%.2f)", route, confidence)

        except Exception as exc:
            fallback = _fallback_route(emotion)
            logger.warning("Routing failed, defaulting to %s: %s", fallback, exc)
            state["route"] = fallback
            state["routing_confidence"] = 0.5

        yield adk_types.Content(parts=[])
        

routing_agent = RoutingAgent(name="routing_agent")