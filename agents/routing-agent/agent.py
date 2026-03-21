import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types as adk_types

from services.llm_service import generate_json

logger = logging.getLogger(__name__)

VALID_ROUTES = {"casual", "therapy", "task", "music", "alert"}
DEFAULT_ROUTE = "casual"
DEFAULT_CONFIDENCE = 0.5
EMPTY_TRANSCRIPT_CONFIDENCE = 1.0
MAX_TRANSCRIPT_CHARS = 300

ROUTING_SYSTEM = """You are a routing classifier for BEAN, a mental health companion robot.
Classify the user's message into exactly one route.

Routes:
- casual: general conversation, small talk, greetings, sharing how their day was
- therapy: sadness, distress, anxiety, loneliness, emotional pain, comfort-seeking
- task: reminders, to-dos, scheduling, planning, task management
- music: requests to play, stop, pause, resume, or change music/songs
- alert: explicit self-harm language, suicidal intent, crisis, immediate safety risk

Rules:
- Return exactly one route
- When in doubt between casual and therapy, choose therapy
- Only choose alert if the language shows clear, explicit self-harm or suicide risk
- Use the detected emotion and recent routing distribution as supporting context
- Do not explain your answer

Respond ONLY with valid JSON:
{"route": "casual|therapy|task|music|alert", "confidence": 0.0}"""

ROUTING_PROMPT = """Classify the current user message.

User message:
{user_text}

Detected emotion: {emotion}
Turn number: {turn_number}
Recent route distribution: {route_distribution}
"""


def _safe_str(value: Any, default: str = "") -> str:
    """Return a safe string value."""
    if value is None:
        return default
    if isinstance(value, str):
        return value
    return str(value)


def _safe_int(value: Any, default: int = 0) -> int:
    """Return a safe integer value."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_route_distribution(value: Any) -> dict[str, int]:
    """Ensure route distribution is always a simple dict."""
    if not isinstance(value, dict):
        return {}

    cleaned: dict[str, int] = {}
    for key, count in value.items():
        key_str = _safe_str(key).strip()
        try:
            cleaned[key_str] = int(count)
        except (TypeError, ValueError):
            cleaned[key_str] = 0
    return cleaned


def _normalize_route(value: Any) -> str:
    """Validate and normalize the returned route."""
    route = _safe_str(value, DEFAULT_ROUTE).strip().lower()
    if route not in VALID_ROUTES:
        logger.warning(
            "Routing returned invalid route '%s'; defaulting to '%s'",
            route,
            DEFAULT_ROUTE,
        )
        return DEFAULT_ROUTE
    return route


def _normalize_confidence(value: Any, default: float = DEFAULT_CONFIDENCE) -> float:
    """Convert confidence to a bounded float between 0.0 and 1.0."""
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default

    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


class RoutingAgent(BaseAgent):
    """Lightweight routing classifier using the cheap routing model tier."""

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[adk_types.Content, None]:
        state = ctx.session.state

        # Read required session state values safely.
        user_text = _safe_str(state.get("current_transcript", ""))
        emotion = _safe_str(state.get("current_emotion", "neutral"), "neutral")
        turn_number = _safe_int(state.get("turn_count", 0), 0)
        route_distribution = _safe_route_distribution(
            state.get("route_distribution", {})
        )

        # Requirement from checklist:
        # Empty transcript should default to casual without calling the LLM.
        if not user_text.strip():
            state["route"] = DEFAULT_ROUTE
            state["routing_confidence"] = EMPTY_TRANSCRIPT_CONFIDENCE
            logger.debug(
                "Empty transcript detected; defaulted route to '%s'", DEFAULT_ROUTE
            )
            yield adk_types.Content(parts=[])
            return

        # Truncate transcript to avoid passing too much text to routing.
        truncated_text = user_text.strip()[:MAX_TRANSCRIPT_CHARS]

        prompt = ROUTING_PROMPT.format(
            user_text=truncated_text,
            emotion=emotion,
            turn_number=turn_number,
            route_distribution=json.dumps(route_distribution, ensure_ascii=False),
        )

        try:
            result = await generate_json(
                task="routing",
                prompt=prompt,
                system=ROUTING_SYSTEM,
            )

            if not isinstance(result, dict):
                raise ValueError("generate_json returned a non-dict response")

            route = _normalize_route(result.get("route"))
            confidence = _normalize_confidence(
                result.get("confidence"), DEFAULT_CONFIDENCE
            )

            state["route"] = route
            state["routing_confidence"] = confidence

            logger.info("Routing decision: route=%s confidence=%.2f", route, confidence)

        except Exception as exc:
            # Requirement from checklist:
            # Malformed / failed response must fall back to casual without raising.
            logger.warning("Routing failed; defaulting to '%s': %s", DEFAULT_ROUTE, exc)
            state["route"] = DEFAULT_ROUTE
            state["routing_confidence"] = DEFAULT_CONFIDENCE

        # Routing agent writes to state only; it does not emit a user-facing message.
        yield adk_types.Content(parts=[])


routing_agent = RoutingAgent(name="routing_agent")
