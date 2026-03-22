"""BEAN AI — Music Agent.

Controls music playback on the BEAN robot via WebSocket commands.
Maps natural language requests to SD card folder names and commands.

ADK Type: LlmAgent (Gemini Flash) with Music FunctionTools.
"""

import logging
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from services.llm_service import generate_json
from shared.enums import MusicAction

logger = logging.getLogger(__name__)

# ── Genre → SD card folder mapping ───────────────────────────────────────────

GENRE_FOLDERS: dict[str, str] = {
    "calm": "calm",
    "relaxing": "calm",
    "sleep": "calm",
    "happy": "happy",
    "upbeat": "happy",
    "energetic": "happy",
    "sad": "sad",
    "melancholy": "sad",
    "lo-fi": "lofi",
    "lofi": "lofi",
    "study": "lofi",
    "focus": "lofi",
    "nature": "nature",
    "ambient": "nature",
    "classical": "classical",
    "piano": "classical",
}

MUSIC_SYSTEM = """You are BEAN's music controller.
Parse the user's music request and extract the command details.

Available commands: play_music, stop_music, next_track, pause_music, resume_music, set_volume

Available genres: calm, happy, sad, lofi, nature, classical

Respond ONLY with this exact JSON — no markdown, no preamble:
{
  "action": "play_music|stop_music|next_track|pause_music|resume_music|set_volume",
  "genre": "genre name or null",
  "volume": 0-100 or null
}"""


class MusicAgent(BaseAgent):
    """Music playback control agent for BEAN robot."""

    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        transcript = ctx.session.state.get("current_transcript", "")
        emotion = ctx.session.state.get("current_emotion", "neutral")

        if not transcript:
            response = "What music would you like to hear?"
            ctx.session.state["response_text"] = response
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"response_text": response}),
            )
            return

        # ── Parse music intent ──
        try:
            parsed = await generate_json(
                task="music_selection",
                prompt=f"User said: {transcript}\nCurrent emotion: {emotion}",
                system=MUSIC_SYSTEM,
            )
        except Exception as exc:
            logger.error("Music intent parsing failed: %s", exc)
            response = "I couldn't understand that music request. Try saying 'play calm music' or 'stop music'."
            ctx.session.state["response_text"] = response
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"response_text": response}),
            )
            return

        action_str = parsed.get("action", "play_music")
        genre = parsed.get("genre")
        volume = parsed.get("volume")

        # ── Auto-select genre based on emotion if not specified ──
        if action_str == "play_music" and not genre:
            genre = self._emotion_to_genre(emotion)

        # ── Build music command for ESP32 ──
        music_command = self._build_command(action_str, genre, volume)
        ctx.session.state["music_command"] = music_command

        # ── Build response text ──
        response = self._build_response(action_str, genre, volume)
        ctx.session.state["response_text"] = response

        logger.info(
            "MusicAgent: action=%s genre=%s volume=%s",
            action_str,
            genre,
            volume,
        )

        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={
                    "response_text": response,
                    "music_command": music_command,
                }
            ),
        )

    def _emotion_to_genre(self, emotion: str) -> str:
        """Map current emotion to appropriate music genre."""
        emotion_genre_map = {
            "sad": "calm",
            "fearful": "calm",
            "angry": "calm",
            "disgusted": "calm",
            "happy": "happy",
            "surprised": "happy",
            "calm": "lofi",
            "neutral": "lofi",
        }
        return emotion_genre_map.get(emotion, "calm")

    def _build_command(
        self,
        action_str: str,
        genre: str | None,
        volume: int | None,
    ) -> dict:
        """Build the music command dict for ESP32."""
        folder = None
        if genre:
            folder = GENRE_FOLDERS.get(genre.lower(), genre.lower())

        return {
            "type": action_str,
            "genre_folder": folder,
            "shuffle": True,
            "volume": volume,
        }

    def _build_response(
        self,
        action_str: str,
        genre: str | None,
        volume: int | None,
    ) -> str:
        """Build a friendly response message."""
        if action_str == "play_music":
            if genre:
                return f"Playing some {genre} music for you 🎵"
            return "Playing music for you 🎵"
        elif action_str == "stop_music":
            return "Stopping the music."
        elif action_str == "pause_music":
            return "Music paused."
        elif action_str == "resume_music":
            return "Music resumed 🎵"
        elif action_str == "next_track":
            return "Skipping to the next track 🎵"
        elif action_str == "set_volume":
            if volume is not None:
                return f"Setting volume to {volume}."
            return "Adjusting the volume."
        return "Got it!"


# ── Singleton ─────────────────────────────────────────────────────────────────
music_agent = MusicAgent(name="music_agent")
