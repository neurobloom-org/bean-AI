"""BEAN AI — Music Agent.

Parses the user's music intent and writes action/mood to session state.
The actual audio streaming is handled by MusicPlayer in the WebSocket handler
so the ESP32 receives audio binary frames directly — no TTS for music audio.

This agent does NOT stream audio and does NOT build ESP32 JSON commands.
It only answers: "what action did the user want, and for which mood?"

Session state written (via EventActions state_delta):
    response_text   — friendly spoken confirmation (read by TTS)
    music_action    — play_music | stop_music | next_track | pause_music | resume_music | set_volume
    music_mood      — calm | happy | sad | lofi | nature | classical (or None)
    music_volume    — 0-100 (or None)

The WebSocket handler reads these after the turn and drives MusicPlayer.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from typing import Final

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from services.llm_service import generate_json
from services.music_service import VALID_MOODS

logger = logging.getLogger(__name__)

DEFAULT_MOOD: Final[str] = "calm"

_EMOTION_MOOD_MAP: Final[dict[str, str]] = {
    "sad": "calm",
    "fearful": "calm",
    "angry": "calm",
    "disgusted": "calm",
    "happy": "happy",
    "surprised": "happy",
    "calm": "lofi",
    "neutral": "calm",
}

_MUSIC_SYSTEM: Final[str] = """You are BEAN's music controller.
Parse the user's music request and output structured intent.

Available actions:
  play_music   — user wants music to start
  stop_music   — user wants music to stop ("stop", "turn off", "quiet", "silence")
  next_track   — skip to next song ("next", "skip", "another one")
  pause_music  — pause music temporarily
  resume_music — resume paused music
  set_volume   — change volume only

Available moods: calm, happy, sad, lofi, nature, classical

Rules:
- If user says "play something" with no mood → set mood to null (backend defaults to calm).
- If user says "stop", "turn off music" → stop_music.
- If user says "louder"/"quieter" → set_volume; do NOT also set play_music.
- volume is 0-100 or null.
- mood is always lowercase or null.

Respond ONLY with this exact JSON — no markdown, no extra text:
{
  "action": "play_music|stop_music|next_track|pause_music|resume_music|set_volume",
  "mood": "calm|happy|sad|lofi|nature|classical|null",
  "volume": 0-100 or null
}"""


class MusicAgent(BaseAgent):
    """Intent parser for music requests."""

    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        transcript: str = ctx.session.state.get("current_transcript", "").strip()
        emotion: str = ctx.session.state.get("current_emotion", "neutral")
        playing_mood: str | None = ctx.session.state.get("music_playing_mood")
        playing_volume: int = int(ctx.session.state.get("music_playing_volume", 50))

        if not transcript:
            response = (
                "What music would you like? I can play calm, happy, lo-fi, "
                "nature, or classical music."
            )
            yield _make_event({"response_text": response})
            return

        try:
            parsed = await generate_json(
                task="music_selection",
                prompt=(
                    f"User said: {transcript}\n"
                    f"Emotion: {emotion}\n"
                    f"Currently playing: {playing_mood or 'nothing'}\n"
                    f"Current volume: {playing_volume}"
                ),
                system=_MUSIC_SYSTEM,
            )
        except Exception as exc:
            logger.error("Music intent parsing failed: %s", exc)
            yield _make_event({
                "response_text": (
                    "I couldn't understand that music request. "
                    "Try saying 'play calm music' or 'stop the music'."
                )
            })
            return

        action: str = (parsed.get("action") or "play_music").strip().lower()
        raw_mood: str | None = parsed.get("mood")
        volume: int | None = parsed.get("volume")

        _valid_actions = {
            "play_music", "stop_music", "next_track",
            "pause_music", "resume_music", "set_volume",
        }
        if action not in _valid_actions:
            action = "play_music"

        # ── Resolve mood ──────────────────────────────────────────────────────
        resolved_mood: str | None = None
        if action in ("play_music", "next_track"):
            if raw_mood and raw_mood.lower() in VALID_MOODS:
                resolved_mood = raw_mood.lower()
            else:
                resolved_mood = _EMOTION_MOOD_MAP.get(emotion, DEFAULT_MOOD)

        # ── Resolve volume ────────────────────────────────────────────────────
        resolved_volume: int | None = None
        if action == "set_volume":
            if volume is not None:
                resolved_volume = max(0, min(100, int(volume)))
            else:
                low = transcript.lower()
                if any(w in low for w in ("louder", "up", "more", "higher")):
                    resolved_volume = min(100, playing_volume + 20)
                else:
                    resolved_volume = max(0, playing_volume - 20)

        response = _build_response(action, resolved_mood, resolved_volume, playing_mood)

        logger.info(
            "MusicAgent: action=%s mood=%s volume=%s [was_playing=%s]",
            action, resolved_mood, resolved_volume, playing_mood,
        )

        yield _make_event({
            "response_text": response,
            "music_action": action,
            "music_mood": resolved_mood,
            "music_volume": resolved_volume,
        })


def _make_event(state_delta: dict) -> Event:
    return Event(author="music_agent", actions=EventActions(state_delta=state_delta))


def _build_response(
    action: str,
    mood: str | None,
    volume: int | None,
    playing_mood: str | None,
) -> str:
    if action == "play_music":
        return f"Playing some {mood} music for you." if mood else "Playing music for you."
    elif action == "stop_music":
        return f"Stopping the {playing_mood} music." if playing_mood else "Stopping the music."
    elif action == "pause_music":
        return "Music paused."
    elif action == "resume_music":
        return "Music resumed."
    elif action == "next_track":
        m = mood or playing_mood
        return f"Skipping to the next {m} song." if m else "Skipping to the next track."
    elif action == "set_volume":
        if volume == 0:
            return "Volume set to zero."
        elif volume and volume >= 80:
            return "Volume turned up!"
        elif volume:
            return f"Volume set to {volume}."
        return "Volume adjusted."
    return "Got it."


music_agent = MusicAgent(name="music_agent")