"""BEAN AI v1 — ESP32 WebSocket handler.

Real-time audio pipeline:
  ESP32 → PCM16 audio → Deepgram STT → Orchestrator → ElevenLabs TTS → ESP32
  ESP32 ← Supabase Storage binary chunks ← MusicPlayer (background task)

Music streaming protocol (binary + JSON over same WebSocket):
  Server → ESP32:
    {"type": "music_start",   "song_id": "...", "title": "...", "mood": "calm"}
    <binary bytes: MP3/OGG audio chunks, ~8 KB each>
    <binary bytes: ...>
    {"type": "music_end"}           — song finished, looping to next
    {"type": "music_stopped"}       — user said stop / session ended

  Server → ESP32 (volume):
    {"type": "music_volume", "volume": 70}

  ESP32 → Server (optional feedback):
    {"type": "music_status", "status": "playing"|"stopped", "genre": "calm"}

TTS and music never overlap:
  When TTS is sending binary audio, MusicPlayer pauses (checks tts_active flag
  before each chunk). The ESP32 speaker hears only one stream at a time.

MusicPlayer lifecycle per session:
  play(mood)   → cancel any current task → start new looping task
  stop()       → set stop_event → cancel task → send music_stopped
  next_track() → set stop_event on current song → restart same mood
  set_volume() → send {"type": "music_volume", "volume": N} immediately
  pause/resume → set/clear paused flag (ESP32 handles buffer drain)
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from agents.orchestrator.agent import orchestrator
from api.middleware.auth_middleware import extract_token_from_websocket
from services.deepgram_service import DeepgramConnection
from services.elevenlabs_service import stream_tts_to_websocket
from services.music_service import pick_song, stream_song_chunks
from services.privacy_service import privacy_service
from services.supabase_client import get_service_client
from shared.config import get_settings
from shared.schemas import utcnow

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Active session registry ────────────────────────────────────────────────────
_active_sessions: dict[str, dict] = {}
_WS_BACKGROUND_TASKS: set[asyncio.Task] = set()


def _fire_and_forget(coro) -> None:
    task = asyncio.create_task(coro)
    _WS_BACKGROUND_TASKS.add(task)
    task.add_done_callback(_WS_BACKGROUND_TASKS.discard)


# ─────────────────────────────────────────────────────────────────────────────
# MusicPlayer
# ─────────────────────────────────────────────────────────────────────────────


class MusicPlayer:
    """Per-session music streaming controller."""

    def __init__(self, session: "BEANSession") -> None:
        self._session = session
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._song_stop_event = asyncio.Event()
        self._mood: str | None = None
        self._paused: bool = False

    @property
    def is_playing(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def mood(self) -> str | None:
        return self._mood

    async def play(self, mood: str) -> None:
        """Start (or restart) looping music for the given mood."""
        await self._cancel_task()
        self._stop_event.clear()
        self._song_stop_event.clear()
        self._paused = False
        self._mood = mood
        self._task = asyncio.create_task(
            self._run_loop(mood),
            name=f"music-{self._session.session_id[:8]}",
        )

    async def stop(self) -> None:
        """Stop music entirely. Sends music_stopped to ESP32."""
        self._mood = None
        self._stop_event.set()
        task_was_running = self._task is not None and not self._task.done()
        await self._cancel_task()
        if not task_was_running:
            await self._session.send_json({"type": "music_stopped"})

    async def next_track(self) -> None:
        """Skip current song, stay in same mood."""
        if self._mood and self.is_playing:
            self._song_stop_event.set()
        elif self._mood:
            await self.play(self._mood)

    async def set_volume(self, volume: int) -> None:
        """Send volume command to ESP32 immediately."""
        volume = max(0, min(100, volume))
        await self._session.send_json({"type": "music_volume", "volume": volume})

    def pause(self) -> None:
        """Pause chunk sending (used while TTS is active)."""
        self._paused = True

    def resume_playback(self) -> None:
        """Resume chunk sending after TTS ends."""
        self._paused = False

    async def cleanup(self) -> None:
        """Call on session disconnect — cancel gracefully."""
        self._stop_event.set()
        await self._cancel_task()

    async def _cancel_task(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await asyncio.wait_for(self._task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._task = None

    async def _run_loop(self, mood: str) -> None:
        """Loop songs of the given mood until stop_event is set."""
        try:
            while not self._stop_event.is_set():
                song = await pick_song(self._session.user_id, mood)

                if not song:
                    logger.warning(
                        "No songs available for mood=%s user=%s",
                        mood,
                        self._session.user_id[:8],
                    )
                    await self._session.send_json({
                        "type": "music_unavailable",
                        "mood": mood,
                        "message": f"No {mood} songs available yet. Upload some!",
                    })
                    break

                self._song_stop_event.clear()

                finished = await self._stream_one_song(song)
                if not finished:
                    break

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error(
                "MusicPlayer loop error [session=%s]: %s",
                self._session.session_id[:8],
                exc,
            )
        finally:
            await self._session.send_json({"type": "music_stopped"})
            logger.info(
                "Music stopped [session=%s mood=%s]",
                self._session.session_id[:8],
                mood,
            )

    async def _stream_one_song(self, song: dict) -> bool:
        """Stream a single song. Returns True if completed normally, False if stopped."""
        song_id = str(song.get("id", ""))
        title = song.get("title", "Music")
        storage_path = song.get("storage_path", "")

        if not storage_path:
            logger.warning("Song %s has no storage_path — skipping", song_id[:8])
            return True

        await self._session.send_json({
            "type": "music_start",
            "song_id": song_id,
            "title": title,
            "mood": song.get("mood", "calm"),
        })

        logger.debug(
            "Streaming song: id=%s title=%r path=%s session=%s",
            song_id[:8],
            title,
            storage_path,
            self._session.session_id[:8],
        )

        try:
            async for chunk in stream_song_chunks(storage_path):
                if self._stop_event.is_set():
                    return False

                if self._song_stop_event.is_set():
                    await self._session.send_json({
                        "type": "music_end",
                        "song_id": song_id,
                    })
                    return True

                while self._session.tts_active:
                    await asyncio.sleep(0.05)
                    if self._stop_event.is_set():
                        return False

                await self._session.send_bytes(chunk)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Song stream error: id=%s error=%s [session=%s]",
                song_id[:8],
                exc,
                self._session.session_id[:8],
            )
            return True

        await self._session.send_json({
            "type": "music_end",
            "song_id": song_id,
        })

        _fire_and_forget(_increment_play_count(song_id))

        return True


# ─────────────────────────────────────────────────────────────────────────────
# BEANSession
# ─────────────────────────────────────────────────────────────────────────────


class BEANSession:
    """Per-connection in-memory state (persisted to Supabase at turn end)."""

    __slots__ = (
        "user_id",
        "session_id",
        "websocket",
        "transcript_buffer",
        "current_emotion",
        "emotion_trend",
        "turn_count",
        "route_distribution",
        "is_processing",
        "is_minor",
        "deepgram",
        "music_player",
        "tts_active",
        "music_playing_mood",
        "music_playing_volume",
        "pending_reminder",
    )

    def __init__(self, user_id: str, session_id: str, websocket: WebSocket) -> None:
        self.user_id = user_id
        self.session_id = session_id
        self.websocket = websocket

        self.transcript_buffer: str = ""
        self.current_emotion: str = "neutral"
        self.emotion_trend: list[str] = []
        self.turn_count: int = 0
        self.route_distribution: dict[str, int] = {}
        self.is_processing: bool = False
        self.is_minor: bool = False
        self.deepgram: DeepgramConnection | None = None

        self.tts_active: bool = False
        self.music_playing_mood: str | None = None
        self.music_playing_volume: int = 50
        self.pending_reminder: dict | None = None

        self.music_player = MusicPlayer(self)

    def add_emotion(self, emotion: str) -> None:
        self.current_emotion = emotion
        self.emotion_trend.append(emotion)
        if len(self.emotion_trend) > 10:
            self.emotion_trend = self.emotion_trend[-10:]

    async def send_json(self, data: dict) -> None:
        try:
            await self.websocket.send_json(data)
        except Exception as exc:
            logger.debug("WS send_json failed [session=%s]: %s", self.session_id[:8], exc)

    async def send_bytes(self, data: bytes) -> None:
        try:
            await self.websocket.send_bytes(data)
        except Exception as exc:
            logger.debug("WS send_bytes failed [session=%s]: %s", self.session_id[:8], exc)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Main WebSocket endpoint for the ESP32 BEAN robot."""
    raw_protocol = websocket.headers.get("Sec-WebSocket-Protocol", "")
    protocols = [p.strip() for p in raw_protocol.split(",") if p.strip()]
    chosen = next((p for p in protocols if p.startswith("bearer.")), None)
    await websocket.accept(subprotocol=chosen)

    device_id = (
        websocket.query_params.get("device_id")
        or websocket.headers.get("x-device-id", "").strip()
    )

    if device_id:
        try:
            # FIX Bug 7: removed duplicate inner import of get_service_client.
            # The module-level import at the top of this file covers it.
            client = await get_service_client()
            result = (
                await client.table("devices")
                .select("user_id")
                .eq("device_id", device_id)
                .maybe_single()
                .execute()
            )
            if not result or not result.data:
                logger.warning("Unknown device_id=%s — closing WebSocket", device_id[:12])
                await websocket.close(code=4001)
                return
            user_id: str = str(result.data["user_id"])
            _fire_and_forget(
                client.table("devices")
                .update({"last_seen_at": utcnow().isoformat()})
                .eq("device_id", device_id)
                .execute()
            )
        except Exception as exc:
            logger.error("Device auth error device_id=%s: %s", device_id[:12], exc)
            await websocket.close(code=4001)
            return
    else:
        token = extract_token_from_websocket(websocket)
        if not token:
            await websocket.close(code=4001)
            return
        try:
            from api.middleware.auth_middleware import decode_supabase_jwt

            payload = await decode_supabase_jwt(token)
            user_id = payload["sub"]
        except Exception as exc:
            logger.warning("JWT auth failed: %s", exc)
            await websocket.close(code=4003)
            return

    session_id = str(uuid.uuid4())
    session = BEANSession(user_id=user_id, session_id=session_id, websocket=websocket)

    await _load_user_flags(session)

    try:
        await _create_session_record(user_id, session_id)
    except Exception as exc:
        logger.error("Failed to create session record: %s", exc)
        await websocket.close(code=1011, reason="Session initialisation failed")
        return

    _active_sessions[session_id] = {
        "session_obj": session,
        "battery_level": None,
        "wifi_rssi": None,
        "user_id": user_id,
        "websocket": websocket,
    }

    logger.info(
        "WebSocket connected: user=%s session=%s is_minor=%s",
        user_id[:8], session_id[:8], session.is_minor,
    )
    await session.send_json({"type": "connected", "session_id": session_id})

    try:
        session.deepgram = DeepgramConnection(
            on_transcript=lambda t: _handle_transcript(session, t),
            on_utterance_end=lambda: _handle_utterance_end(session),
        )
        await session.deepgram.connect()
    except Exception as exc:
        logger.error("Deepgram connection failed: %s", exc)
        await session.send_json(
            {"type": "error", "code": "stt_unavailable", "message": str(exc)}
        )

    settings = get_settings()
    reminder_task = asyncio.create_task(
        _poll_reminders(session), name=f"reminders-{session_id[:8]}"
    )
    keepalive_task = asyncio.create_task(
        _keepalive_ping(session, interval=settings.ws_ping_interval_seconds),
        name=f"keepalive-{session_id[:8]}",
    )

    try:
        while True:
            message = await websocket.receive()

            if message.get("bytes"):
                audio_bytes: bytes = message["bytes"]
                if len(audio_bytes) > settings.ws_max_audio_frame_bytes:
                    logger.warning(
                        "Audio frame too large (%d bytes) [session=%s]",
                        len(audio_bytes), session_id[:8],
                    )
                    continue
                if session.deepgram and session.deepgram.is_connected:
                    await session.deepgram.send_audio(audio_bytes)

            elif message.get("text"):
                try:
                    data = json.loads(message["text"])
                    await _handle_control_message(session, data)
                except json.JSONDecodeError:
                    logger.warning(
                        "Invalid JSON from ESP32 [session=%s]: %.100s",
                        session_id[:8], message["text"],
                    )

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected: user=%s session=%s turns=%d",
            user_id[:8], session_id[:8], session.turn_count,
        )
    except Exception as exc:
        logger.error("WebSocket error [session=%s]: %s", session_id[:8], exc)
    finally:
        reminder_task.cancel()
        keepalive_task.cancel()
        _active_sessions.pop(session_id, None)
        await session.music_player.cleanup()
        await _cleanup_session(session)


# ─────────────────────────────────────────────────────────────────────────────
# STT handlers
# ─────────────────────────────────────────────────────────────────────────────


async def _handle_transcript(session: BEANSession, transcript_result) -> None:
    if not transcript_result.is_final:
        await session.send_json(
            {"type": "transcript_partial", "text": transcript_result.text}
        )
        return
    text = transcript_result.text.strip()
    if not text:
        return
    session.transcript_buffer += (" " + text) if session.transcript_buffer else text
    await session.send_json({
        "type": "transcript_final",
        "text": text,
        "confidence": round(transcript_result.confidence, 2),
    })


async def _handle_utterance_end(session: BEANSession) -> None:
    if session.music_playing_mood is not None:
        await session.music_player.stop()
        session.music_playing_mood = None
        logger.info("Music stopped for incoming utterance [session=%s]", session.session_id[:8])

    if session.is_processing:
        return
    user_text = session.transcript_buffer.strip()
    if not user_text:
        return

    session.is_processing = True
    session.transcript_buffer = ""
    try:
        await _process_turn(session, user_text)
    except Exception as exc:
        logger.error("Turn processing failed [session=%s]: %s", session.session_id[:8], exc)
        await session.send_json({
            "type": "error",
            "code": "processing_failed",
            "message": "Something went wrong — please try again.",
        })
    finally:
        session.is_processing = False


# ─────────────────────────────────────────────────────────────────────────────
# Core turn
# ─────────────────────────────────────────────────────────────────────────────


async def _process_turn(session: BEANSession, user_text: str) -> None:
    """Run the full orchestrator pipeline for one conversation turn."""
    session.turn_count += 1
    turn_id = str(uuid.uuid4())

    recent = await privacy_service.get_recent_transcript(
        session_id=session.session_id, max_turns=10
    )

    session_state: dict = {
        "user_id": session.user_id,
        "session_id": session.session_id,
        "current_transcript": user_text,
        "current_emotion": session.current_emotion,
        "emotion_confidence": "0.7",
        "recent_emotions": json.dumps([[e, 0.7] for e in session.emotion_trend[-5:]]),
        "turn_count": session.turn_count,
        "route_distribution": session.route_distribution,
        "recent_transcript": recent,
        "is_minor": session.is_minor,
        "alert_dispatched": "false",
        "pending_reminder": session.pending_reminder,
        "music_playing_mood": session.music_playing_mood,
        "music_playing_volume": session.music_playing_volume,
    }

    response_text = ""
    runner = InMemoryRunner(agent=orchestrator)
    runner_session = await runner.session_service.create_session(
        app_name=runner.app_name,
        user_id=session.user_id,
        state=session_state,
    )

    async for event in runner.run_async(
        user_id=session.user_id,
        session_id=runner_session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=user_text)],
        ),
    ):
        if event.content and event.content.parts:
            response_text = event.content.parts[0].text or ""

    updated_session = await runner.session_service.get_session(
        app_name=runner.app_name,
        user_id=session.user_id,
        session_id=runner_session.id,
    )
    if updated_session:
        session_state = dict(updated_session.state)

    if session.pending_reminder:
        session.pending_reminder = None

    route: str = session_state.get("route", "casual")
    if not response_text:
        response_text = session_state.get("response_text", "")

    if not response_text:
        logger.warning(
            "Orchestrator returned no text [session=%s route=%s]",
            session.session_id[:8], route,
        )
        return

    session.route_distribution[route] = session.route_distribution.get(route, 0) + 1

    await session.send_json({
        "type": "response_text",
        "text": response_text,
        "route": route,
        "turn_id": turn_id,
    })

    post_alert = session_state.get("post_alert_message")
    if post_alert:
        await session.send_json({"type": "alert_notification", "message": post_alert})

    session.tts_active = True
    try:
        await stream_tts_to_websocket(
            text=response_text.strip(),
            websocket=session.websocket,
            turn_id=turn_id,
        )
    except Exception as exc:
        logger.error(
            "TTS stream failed [session=%s turn=%s]: %s",
            session.session_id[:8], turn_id[:8], exc,
        )
        await session.send_json(
            {"type": "error", "code": "tts_failed", "message": "Audio unavailable."}
        )
    finally:
        session.tts_active = False

    if route == "music":
        await _handle_music_action(session, session_state)


# ─────────────────────────────────────────────────────────────────────────────
# Music action handler
# ─────────────────────────────────────────────────────────────────────────────


async def _handle_music_action(session: BEANSession, session_state: dict) -> None:
    """Process music_action from the MusicAgent's session state."""
    action: str = str(session_state.get("music_action") or "play_music")
    mood: str = str(session_state.get("music_mood") or "calm")
    volume: int | None = session_state.get("music_volume")

    logger.info(
        "Music action: %s mood=%s volume=%s [session=%s]",
        action, mood, volume, session.session_id[:8],
    )

    if action == "play_music":
        await session.music_player.play(mood)
        session.music_playing_mood = mood

    elif action == "stop_music":
        await session.music_player.stop()
        session.music_playing_mood = None

    elif action == "next_track":
        await session.music_player.next_track()

    elif action == "pause_music":
        session.music_player.pause()
        await session.send_json({"type": "music_paused"})

    elif action == "resume_music":
        session.music_player.resume_playback()
        await session.send_json({"type": "music_resumed"})
        if not session.music_player.is_playing and session.music_playing_mood:
            await session.music_player.play(session.music_playing_mood)

    elif action == "set_volume":
        if volume is not None:
            await session.music_player.set_volume(volume)
            session.music_playing_volume = volume


# ─────────────────────────────────────────────────────────────────────────────
# Control message dispatcher
# ─────────────────────────────────────────────────────────────────────────────


async def _handle_control_message(session: BEANSession, data: dict) -> None:
    """Route incoming ESP32 JSON control messages."""
    msg_type = data.get("type", "")

    if msg_type == "ping":
        await session.send_json({"type": "pong", "timestamp": utcnow().isoformat()})

    elif msg_type == "emotion_result":
        emotion = data.get("emotion", "neutral")
        confidence = round(float(data.get("confidence", 0.0)), 2)
        session.add_emotion(emotion)
        _fire_and_forget(_store_emotion_event(session, emotion, confidence))

    elif msg_type == "music_status":
        if data.get("status") == "stopped":
            session.music_playing_mood = None

    elif msg_type == "robot_status":
        battery = data.get("battery_level")
        rssi = data.get("wifi_rssi")
        logger.debug(
            "Robot status: battery=%s wifi=%s session=%s",
            battery,
            rssi,
            session.session_id[:8],
        )
        if session.session_id in _active_sessions:
            _active_sessions[session.session_id].update({"battery_level": battery, "wifi_rssi": rssi})

    elif msg_type == "end_session":
        await _cleanup_session(session)

    else:
        logger.debug("Unknown control message '%s' [session=%s]", msg_type, session.session_id[:8])


# ─────────────────────────────────────────────────────────────────────────────
# Background helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _keepalive_ping(session: BEANSession, interval: int = 20) -> None:
    while True:
        try:
            await asyncio.sleep(interval)
            await session.send_json({"type": "ping", "timestamp": utcnow().isoformat()})
        except asyncio.CancelledError:
            break
        except Exception:
            break


async def _poll_reminders(session: BEANSession) -> None:
    from services.redis_service import redis_get, redis_delete

    while True:
        try:
            await asyncio.sleep(15)
            redis_key = f"pending_reminder:{session.user_id}"
            reminder = await redis_get(redis_key)
            if not reminder:
                continue
            await redis_delete(redis_key)
            session.pending_reminder = reminder
            logger.info(
                "Reminder queued for next turn [session=%s task=%s]",
                session.session_id[:8],
                str(reminder.get("task_id", ""))[:8],
            )
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.debug("Reminder poll error [session=%s]: %s", session.session_id[:8], exc)


async def _increment_play_count(song_id: str) -> None:
    """Increment the play_count for a song (best-effort, non-blocking)."""
    try:
        client = await get_service_client()
        await client.rpc("increment_song_play_count", {"p_song_id": song_id}).execute()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Session lifecycle
# ─────────────────────────────────────────────────────────────────────────────


async def _load_user_flags(session: BEANSession) -> None:
    try:
        from services.supabase_client import get_user_profile
        profile = await get_user_profile(session.user_id)
        if profile:
            diagnosis_tags = profile.get("diagnosis_tags", []) or []
            session.is_minor = (
                bool(profile.get("is_minor", False)) or "minor" in diagnosis_tags
            )
    except Exception as exc:
        logger.debug("Failed to load user flags [session=%s]: %s", session.session_id[:8], exc)


async def _create_session_record(user_id: str, session_id: str) -> None:
    try:
        client = await get_service_client()
        await (
            client.table("sessions")
            .insert({
                "id": session_id,
                "user_id": user_id,
                "started_at": utcnow().isoformat(),
                "status": "active",
            })
            .execute()
        )
    except Exception as exc:
        logger.error(
            "Failed to create session record [session=%s user=%s]: %s",
            session_id[:8], user_id[:8], exc,
        )
        raise


async def _store_emotion_event(session: BEANSession, emotion: str, confidence: float) -> None:
    try:
        client = await get_service_client()
        await (
            client.table("emotion_events")
            .insert({
                "user_id": session.user_id,
                "session_id": session.session_id,
                "emotion": emotion,
                "confidence": confidence,
            })
            .execute()
        )
    except Exception as exc:
        logger.debug("Emotion event store failed (non-critical) [session=%s]: %s", session.session_id[:8], exc)


async def _cleanup_session(session: BEANSession) -> None:
    if session.deepgram:
        try:
            await session.deepgram.close()
        except Exception:
            pass
    try:
        client = await get_service_client()
        await (
            client.table("sessions")
            .update({
                "status": "ended",
                "ended_at": utcnow().isoformat(),
                "turn_count": session.turn_count,
                "route_distribution": session.route_distribution,
                "state_json": {},
            })
            .eq("id", session.session_id)
            .execute()
        )

        from shared.conversation_cache import clear_history
        try:
            await clear_history(session.session_id)
        except Exception:
            pass

    except Exception as exc:
        logger.error("Session cleanup DB update failed [session=%s]: %s", session.session_id[:8], exc)
    logger.info(
        "Session ended: user=%s session=%s turns=%d",
        session.user_id[:8], session.session_id[:8], session.turn_count,
    )