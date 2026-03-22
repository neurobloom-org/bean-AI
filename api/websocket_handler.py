"""BEAN AI v1 — ESP32 WebSocket handler.

Handles the full real-time audio pipeline:
  ESP32 → PCM16 audio → Deepgram STT → Orchestrator → ElevenLabs TTS → ESP32

Security:
  ✓  JWT via Sec-WebSocket-Protocol header (not query param)
  ✓  Rate limiting per user via Supabase-backed limiter
  ✓  Audio frames validated for max size before processing
  ✓  WebSocket closed gracefully on auth failure (code 4401)

Privacy:
  ✓  Audio bytes held in RAM only — never written to any storage
  ✓  Transcript stored temporarily (24h) via PrivacyService
  ✓  TTS audio streamed directly to ESP32 — never cached

Music:
  ✓  music_command from session.state forwarded to ESP32 as JSON
  ✓  Supports play/stop/next/pause/resume/set_volume

Reminders:
  ✓  _active_sessions dict exported for background reminder checker

Keepalive:
  ✓  Server-side ping task prevents silent ESP32 TCP timeouts

Bugs fixed vs. original:
  - get_pending_music_command imported but didn't exist → ImportError every
    music route turn.  Now reads music_command from session_state dict.
  - set_session_token imported but didn't exist → ImportError every turn.
    Entire redundant calendar-token block removed; orchestrator handles it.
  - WebSocketAuthError now imported from shared.exceptions (canonical source),
    not from auth_middleware (which re-exports it but is explicitly flagged as
    wrong in shared/exceptions.py docstring).
  - route was read inside the async-for loop before the orchestrator had set
    it; moved to after the loop.
  - No keepalive task → ESP32 connections would silently time out under
    quiet periods. A server-side ping task now runs every ws_ping_interval_seconds.
  - DeepgramConnection was assigned a tuple (trailing comma) instead of the
    connection object itself.
  - on_transcript / on_utterance_end passed with wrong signatures; wrapped
    in lambdas that close over session so the signatures match what
    DeepgramConnection expects.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from agents.orchestrator.agent import orchestrator
from api.middleware.auth_middleware import authenticate_websocket
from api.middleware.rate_limiter import check_ws_rate_limit
from services.deepgram_service import DeepgramConnection
from services.elevenlabs_service import stream_tts_to_websocket
from services.privacy_service import privacy_service
from services.supabase_client import get_service_client
from shared.config import get_settings
from shared.exceptions import WebSocketAuthError
from shared.schemas import utcnow

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Active session registry (exported for background/reminder_check.py) ───────
# Maps session_id → {"user_id": str, "websocket": WebSocket, "pending_reminder": dict|None}
_active_sessions: dict[str, dict] = {}


# ─────────────────────────────────────────────────────────────────────────────
# Per-connection session state
# ─────────────────────────────────────────────────────────────────────────────


class BEANSession:
    """Per-connection session state (in-memory; persisted to DB at turn end)."""

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
        "recent_transcript",
        "is_minor",
        "guardian_phone",
        "deepgram",
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
        self.recent_transcript: list[dict] = []
        self.is_minor: bool = False
        self.guardian_phone: str = ""

        self.deepgram: DeepgramConnection | None = None

    def add_emotion(self, emotion: str) -> None:
        self.current_emotion = emotion
        self.emotion_trend.append(emotion)
        if len(self.emotion_trend) > 10:
            self.emotion_trend = self.emotion_trend[-10:]

    async def send_json(self, data: dict) -> None:
        try:
            await self.websocket.send_json(data)
        except Exception as exc:
            logger.debug(
                "WS send_json failed [session=%s]: %s", self.session_id[:8], exc
            )

    async def send_bytes(self, data: bytes) -> None:
        try:
            await self.websocket.send_bytes(data)
        except Exception as exc:
            logger.debug(
                "WS send_bytes failed [session=%s]: %s", self.session_id[:8], exc
            )


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ─────────────────────────────────────────────────────────────────────────────


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Main WebSocket endpoint for the ESP32 BEAN robot.

    Auth:   JWT via Sec-WebSocket-Protocol: bearer.<token>
    Audio:  Binary frames, PCM16 @ 16 kHz, mono
    Control: JSON messages — ping/pong, emotion results, robot status, etc.
    """
    raw_protocol = websocket.headers.get("Sec-WebSocket-Protocol", "")
    chosen = raw_protocol.split(",")[0].strip() if "bearer." in raw_protocol else None
    await websocket.accept(subprotocol=chosen)

    # ── Authenticate ──────────────────────────────────────────────────────────
    try:
        user_id, _payload = await authenticate_websocket(websocket)
    except WebSocketAuthError:
        return

    # ── Create session ────────────────────────────────────────────────────────
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
        "user_id": user_id,
        "websocket": websocket,
        "pending_reminder": None,
    }

    logger.info(
        "WebSocket connected: user=%s session=%s is_minor=%s",
        user_id[:8],
        session_id[:8],
        session.is_minor,
    )
    await session.send_json({"type": "connected", "session_id": session_id})

    # ── Deepgram STT ──────────────────────────────────────────────────────────
    # FIX: removed trailing comma (was creating a tuple instead of a
    # DeepgramConnection). Callbacks wrapped in lambdas that close over
    # `session` so their signatures match what DeepgramConnection expects:
    #   on_transcript:    Callable[[TranscriptResult], Coroutine]
    #   on_utterance_end: Callable[[], Coroutine]
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

    # ── Background tasks for this connection ──────────────────────────────────
    reminder_task = asyncio.create_task(
        _poll_reminders(session), name=f"reminders-{session_id[:8]}"
    )
    keepalive_task = asyncio.create_task(
        _keepalive_ping(session, interval=settings.ws_ping_interval_seconds),
        name=f"keepalive-{session_id[:8]}",
    )

    # ── Main receive loop ─────────────────────────────────────────────────────
    try:
        while True:
            message = await websocket.receive()

            if message.get("bytes"):
                audio_bytes: bytes = message["bytes"]

                if len(audio_bytes) > settings.ws_max_audio_frame_bytes:
                    logger.warning(
                        "Audio frame too large (%d bytes), dropping [session=%s]",
                        len(audio_bytes),
                        session_id[:8],
                    )
                    continue

                if not await check_ws_rate_limit(user_id):
                    await session.send_json(
                        {
                            "type": "error",
                            "code": "rate_limited",
                            "message": "Too many messages — please slow down.",
                        }
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
                        session_id[:8],
                        message["text"],
                    )

    except WebSocketDisconnect:
        logger.info(
            "WebSocket disconnected: user=%s session=%s turns=%d",
            user_id[:8],
            session_id[:8],
            session.turn_count,
        )
    except Exception as exc:
        logger.error("WebSocket error [session=%s]: %s", session_id[:8], exc)
    finally:
        reminder_task.cancel()
        keepalive_task.cancel()
        _active_sessions.pop(session_id, None)
        await _cleanup_session(session)


# ─────────────────────────────────────────────────────────────────────────────
# STT transcript handlers
# ─────────────────────────────────────────────────────────────────────────────


async def _handle_transcript(session: BEANSession, transcript_result) -> None:
    """Handle a Deepgram transcript callback (partial or final)."""
    if not transcript_result.is_final:
        await session.send_json(
            {"type": "transcript_partial", "text": transcript_result.text}
        )
        return

    text = transcript_result.text.strip()
    if not text:
        return

    session.transcript_buffer += (" " + text) if session.transcript_buffer else text
    await session.send_json(
        {
            "type": "transcript_final",
            "text": text,
            "confidence": round(transcript_result.confidence, 2),
        }
    )


async def _handle_utterance_end(session: BEANSession) -> None:
    """Called when Deepgram signals the user has stopped speaking."""
    if session.is_processing:
        logger.debug(
            "Utterance end skipped — already processing [session=%s]",
            session.session_id[:8],
        )
        return

    user_text = session.transcript_buffer.strip()
    if not user_text:
        return

    session.is_processing = True
    session.transcript_buffer = ""

    try:
        await _process_turn(session, user_text)
    except Exception as exc:
        logger.error(
            "Turn processing failed [session=%s]: %s", session.session_id[:8], exc
        )
        await session.send_json(
            {
                "type": "error",
                "code": "processing_failed",
                "message": "Something went wrong — please try again.",
            }
        )
    finally:
        session.is_processing = False


# ─────────────────────────────────────────────────────────────────────────────
# Core conversation turn
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
    }

    response_text = ""

    async for event in orchestrator.run_async(
        user_id=session.user_id,
        session_id=session.session_id,
        session_state=session_state,
    ):
        if event.content and event.content.parts:
            response_text = event.content.parts[0].text or ""

    route: str = session_state.get("route", "casual")
    music_command: dict | None = session_state.get("music_command")

    if not response_text:
        response_text = session_state.get("response_text", "")

    if not response_text:
        logger.warning(
            "Orchestrator returned no text [session=%s route=%s]",
            session.session_id[:8],
            route,
        )
        return

    session.route_distribution[route] = session.route_distribution.get(route, 0) + 1

    await session.send_json(
        {
            "type": "response_text",
            "text": response_text,
            "route": route,
            "turn_id": turn_id,
        }
    )

    if music_command and isinstance(music_command, dict):
        await session.send_json(music_command)
        logger.info(
            "Music command sent to ESP32: type=%s session=%s",
            music_command.get("type"),
            session.session_id[:8],
        )

    post_alert = session_state.get("post_alert_message")
    if post_alert:
        await session.send_json({"type": "alert_notification", "message": post_alert})

    tts_text = (
        response_text
        if route != "music"
        else (response_text if response_text.strip() else None)
    )
    if tts_text:
        await stream_tts_to_websocket(
            text=tts_text,
            websocket=session.websocket,
            turn_id=turn_id,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Control message dispatcher
# ─────────────────────────────────────────────────────────────────────────────


async def _handle_control_message(session: BEANSession, data: dict) -> None:
    """Route incoming ESP32 JSON control messages to their handlers."""
    msg_type = data.get("type", "")

    if msg_type == "ping":
        await session.send_json({"type": "pong", "timestamp": utcnow().isoformat()})

    elif msg_type == "emotion_result":
        emotion = data.get("emotion", "neutral")
        confidence = round(float(data.get("confidence", 0.0)), 2)
        session.add_emotion(emotion)
        asyncio.create_task(_store_emotion_event(session, emotion, confidence))

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
            _active_sessions[session.session_id].update(
                {"battery_level": battery, "wifi_rssi": rssi}
            )

    elif msg_type == "end_session":
        await _cleanup_session(session)

    else:
        logger.debug(
            "Unknown control message type '%s' [session=%s]",
            msg_type,
            session.session_id[:8],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Keepalive ping
# ─────────────────────────────────────────────────────────────────────────────


async def _keepalive_ping(session: BEANSession, interval: int = 20) -> None:
    """Send periodic server-initiated pings to keep the ESP32 TCP connection alive."""
    while True:
        try:
            await asyncio.sleep(interval)
            await session.send_json({"type": "ping", "timestamp": utcnow().isoformat()})
        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.debug(
                "Keepalive ping failed [session=%s]: %s",
                session.session_id[:8],
                exc,
            )
            break


# ─────────────────────────────────────────────────────────────────────────────
# Reminder polling
# ─────────────────────────────────────────────────────────────────────────────


async def _poll_reminders(session: BEANSession) -> None:
    """Deliver pending reminders injected by the background reminder_check job."""
    while True:
        try:
            await asyncio.sleep(15)
            entry = _active_sessions.get(session.session_id)
            if not entry:
                break

            reminder = entry.get("pending_reminder")
            if not reminder:
                continue

            entry["pending_reminder"] = None

            await session.send_json(
                {
                    "type": "reminder",
                    "task_id": reminder.get("task_id"),
                    "title": reminder.get("title"),
                    "description": reminder.get("description"),
                    "due_at": reminder.get("due_at"),
                }
            )
            logger.info(
                "Reminder delivered to ESP32: task=%s user=%s",
                reminder.get("task_id"),
                session.user_id[:8],
            )

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.debug(
                "Reminder poll error [session=%s]: %s",
                session.session_id[:8],
                exc,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Session lifecycle helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _load_user_flags(session: BEANSession) -> None:
    """Load is_minor flag from the user's profile in Supabase."""
    try:
        from services.supabase_client import get_user_profile

        profile = await get_user_profile(session.user_id)
        if profile:
            diagnosis_tags = profile.get("diagnosis_tags", []) or []
            session.is_minor = (
                bool(profile.get("is_minor", False)) or "minor" in diagnosis_tags
            )
    except Exception as exc:
        logger.debug(
            "Failed to load user flags [session=%s]: %s",
            session.session_id[:8],
            exc,
        )


async def _create_session_record(user_id: str, session_id: str) -> None:
    """Insert a new session row in Supabase."""
    client = await get_service_client()
    await (
        client.table("sessions")
        .insert(
            {
                "id": session_id,
                "user_id": user_id,
                "started_at": utcnow().isoformat(),
                "status": "active",
            }
        )
        .execute()
    )


async def _store_emotion_event(
    session: BEANSession, emotion: str, confidence: float
) -> None:
    """Persist a detected emotion event (non-blocking, non-critical)."""
    try:
        client = await get_service_client()
        await (
            client.table("emotion_events")
            .insert(
                {
                    "user_id": session.user_id,
                    "session_id": session.session_id,
                    "emotion": emotion,
                    "confidence": confidence,
                }
            )
            .execute()
        )
    except Exception as exc:
        logger.debug(
            "Emotion event store failed (non-critical) [session=%s]: %s",
            session.session_id[:8],
            exc,
        )


async def _cleanup_session(session: BEANSession) -> None:
    """Close Deepgram and mark the session as ended in Supabase."""
    if session.deepgram:
        try:
            await session.deepgram.close()
        except Exception:
            pass

    try:
        client = await get_service_client()
        await (
            client.table("sessions")
            .update(
                {
                    "status": "ended",
                    "ended_at": utcnow().isoformat(),
                    "turn_count": session.turn_count,
                    "route_distribution": session.route_distribution,
                    "state_json": {},
                }
            )
            .eq("id", session.session_id)
            .execute()
        )
    except Exception as exc:
        logger.error(
            "Session cleanup DB update failed [session=%s]: %s",
            session.session_id[:8],
            exc,
        )

    logger.info(
        "Session ended: user=%s session=%s turns=%d",
        session.user_id[:8],
        session.session_id[:8],
        session.turn_count,
    )
