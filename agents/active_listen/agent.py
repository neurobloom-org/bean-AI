"""BEAN AI — Active Listen Agent.

Rule-based filler phrase selection based on detected emotion.
Returns pre-cached TTS audio references for immediate playback.

Target latency: <50ms (no LLM calls, no external API calls)
ADK Type: FunctionTool — pure rule-based logic.
"""

import logging
import random

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


# ── ADK FunctionTool wrapper ──────────────────────────────────────────────────
get_filler_phrase_tool = FunctionTool(func=get_filler_phrase)
