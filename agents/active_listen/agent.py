"""BEAN AI — Active Listen Agent.

Rule-based filler phrase selection based on detected emotion.
Returns pre-cached TTS audio references for immediate playback.

Target latency: <50ms (no LLM calls, no external API calls)
ADK Type: Both FunctionTool (legacy) and BaseAgent (used by orchestrator).

The orchestrator calls `active_listen_agent` (the BaseAgent) in the therapy
pipeline as a non-blocking pre-step. It selects a short filler phrase (e.g.
"I hear you...") to fill the latency gap while the therapeutic response is
generated.

FIX: Added `ActiveListenAgent` (BaseAgent subclass) and `active_listen_agent`
singleton. The previous version only exported a FunctionTool, so the
orchestrator's `from agents.active_listen.agent import active_listen_agent`
raised an ImportError, silently degrading every therapy turn.
"""

import logging
import random
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.tools import FunctionTool

from shared.enums import EmotionLabel
from shared.schemas import FillerPhraseResult

logger = logging.getLogger(__name__)

# ── Filler phrases by emotion ─────────────────────────────────────────────────

_PHRASES: dict[str, list[str]] = {
    "sad": [
        "I'm here with you...",
        "Take your time, I'm listening...",
        "I hear you...",
        "That sounds really hard...",
    ],
    "angry": [
        "I understand...",
        "That sounds frustrating...",
        "I'm listening...",
        "Tell me more...",
    ],
    "fearful": [
        "You're safe here...",
        "I'm right here with you...",
        "Take a breath...",
        "I'm listening...",
    ],
    "disgusted": [
        "I hear you...",
        "That sounds really difficult...",
        "I'm here...",
        "Tell me what's going on...",
    ],
    "happy": [
        "That's wonderful!",
        "Tell me more!",
        "I love hearing this!",
        "Go on...",
    ],
    "calm": [
        "Mmm...",
        "I see...",
        "Go on...",
        "Tell me more...",
    ],
    "surprised": [
        "Oh wow...",
        "Really?",
        "Tell me more!",
        "I'm listening...",
    ],
    "neutral": [
        "Hmm, let me think about that...",
        "I'm here...",
        "Go on...",
        "Tell me more...",
    ],
}

# Default phrases when emotion is unknown
_DEFAULT_PHRASES = [
    "Hmm, let me think about that...",
    "I'm here...",
    "Go on...",
    "I'm listening...",
]


def get_filler_phrase(
    emotion: str = "neutral",
    last_phrase: str = "",
) -> dict:
    """Select a context-appropriate filler phrase for Bean.

    Rule-based selection based on detected emotion.
    Avoids repeating the last phrase used.

    Args:
        emotion:     Current detected emotion label.
        last_phrase: Previously used phrase (to avoid repetition).

    Returns:
        FillerPhraseResult as dict with phrase and audio_cache_key.
    """
    # Normalize emotion
    try:
        emotion_label = EmotionLabel(emotion.lower())
        phrases = _PHRASES.get(emotion_label.value, _DEFAULT_PHRASES)
    except (ValueError, AttributeError):
        logger.debug("Unknown emotion '%s', using default phrases", emotion)
        phrases = _DEFAULT_PHRASES

    # Filter out last phrase to avoid repetition
    available = [p for p in phrases if p != last_phrase]
    if not available:
        available = phrases

    phrase = random.choice(available)

    # Build cache key for TTS pre-caching
    safe_key = (
        phrase.lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("!", "")
        .replace("?", "")
    )
    audio_cache_key = f"filler_{emotion}_{safe_key}"

    result = FillerPhraseResult(
        phrase=phrase,
        audio_cache_key=audio_cache_key,
        audio_b64=None,
    )

    logger.debug("Filler phrase selected: '%s' for emotion=%s", phrase, emotion)

    return result.model_dump()


# ── BaseAgent wrapper ─────────────────────────────────────────────────────────


class ActiveListenAgent(BaseAgent):
    """Rule-based filler phrase selector.

    Used by the orchestrator as a non-blocking pre-step in the therapy pipeline.
    Selects a short acknowledgement phrase based on the user's current emotion
    and writes it to session state. The phrase can be played immediately while
    the full therapeutic response is being generated, reducing perceived latency.

    Writes to session state (via state_delta):
        filler_phrase          — the chosen phrase text
        filler_audio_cache_key — Supabase tts_cache lookup key
        last_filler_phrase     — persisted to avoid repetition next turn

    Never raises — any failure falls back silently to empty state values.
    """

    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        emotion: str = str(state.get("current_emotion") or "neutral")
        last_phrase: str = str(state.get("last_filler_phrase") or "")

        try:
            result = get_filler_phrase(emotion=emotion, last_phrase=last_phrase)
            phrase: str = result.get("phrase", "")
            audio_cache_key: str = result.get("audio_cache_key", "")
        except Exception as exc:
            logger.warning("ActiveListenAgent: filler phrase selection failed: %s", exc)
            phrase = ""
            audio_cache_key = ""

        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={
                    "filler_phrase": phrase,
                    "filler_audio_cache_key": audio_cache_key,
                    "last_filler_phrase": phrase,
                }
            ),
        )


# ── Singletons ────────────────────────────────────────────────────────────────

# BaseAgent singleton used by the orchestrator in the therapy pipeline.
active_listen_agent = ActiveListenAgent(name="active_listen_agent")

# FunctionTool singleton kept for any legacy ADK tool-based flows.
get_filler_phrase_tool = FunctionTool(func=get_filler_phrase)
