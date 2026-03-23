"""BEAN AI — STT Agent: Deepgram streaming session manager + fallback FunctionTool.

ARCHITECTURE NOTE
-----------------
Production path (hot path):
    ESP32 binary WebSocket frame -> FastAPI WS handler -> stream_audio_chunk()
    -> DeepgramConnection.send_audio()
    Raw PCM bytes bypass ADK entirely. No base64 overhead on the audio firehose.

Fallback / test path only:
    deepgram_transcribe(session_id, audio_b64) — registered as ADK FunctionTool.
    Used by the orchestrator in controlled/test flows, NOT for live robot audio.

Cloud Run note:
    Session state is in-memory for the current container instance only.
    Reconnects may land on a different instance. The WebSocket handler must
    rebuild session state on reconnect rather than assuming this registry is
    durable across instances.

Privacy:
    Audio bytes flow ESP32 -> FastAPI WS handler -> Deepgram in RAM only.
    They are NEVER written to disk, Supabase, or any storage layer.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from google.adk.tools import FunctionTool

from services.deepgram_service import DeepgramConnection
from shared.exceptions import DeepgramConnectionError
from shared.schemas import TranscriptResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Tuning constants
# ─────────────────────────────────────────────────────────────────────────────

# How long to wait for Deepgram's WebSocket handshake before giving up.
CONNECT_TIMEOUT_SECONDS = 8.0

# How long to wait for a single audio send before treating it as a hung socket.
SEND_TIMEOUT_SECONDS = 2.0

# How long to wait for Deepgram to drain and close cleanly on session end.
CLOSE_TIMEOUT_SECONDS = 3.0

# Deepgram closes idle connections after ~10s without audio or a keepalive.
# We send a keepalive every 4s when the session has been silent for 3.5s.
KEEPALIVE_INTERVAL_SECONDS = 4.0
KEEPALIVE_IDLE_THRESHOLD_SECONDS = 3.5

# The audio path runs at ~50 frames/second. Without rate-limiting, a single
# dropped connection produces thousands of log lines per minute.
DISCONNECTED_LOG_INTERVAL_SECONDS = 5.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-session state
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DeepgramSession:
    """All Deepgram state for one active BEAN session.

    Uses __slots__ (via slots=True) so attribute access on the hot audio
    path is as fast as possible — this object is touched on every frame.
    """

    session_id: str
    connection: DeepgramConnection

    # Background task that sends keepalives during silence.
    keepalive_task: asyncio.Task[None] | None = None

    # Monotonic timestamp of the last audio frame forwarded to Deepgram.
    # Used by the keepalive loop to detect silence.
    last_audio_monotonic: float = field(default_factory=time.monotonic)

    # Monotonic timestamp of the last "connection inactive" log line.
    # Used to suppress repetitive disconnect logs from the hot audio path.
    last_disconnected_log_monotonic: float = 0.0

    @property
    def is_connected(self) -> bool:
        """True only if the underlying Deepgram WebSocket is currently live."""
        return self.connection.is_connected


# ─────────────────────────────────────────────────────────────────────────────
# Module-level registry  (current container instance only — not durable)
# ─────────────────────────────────────────────────────────────────────────────

# One DeepgramSession per active BEAN session.
_active_sessions: dict[str, DeepgramSession] = {}

# Tracks in-flight create_deepgram_connection() tasks so that two concurrent
# callers for the same session_id share one connect attempt instead of racing
# to create two WebSocket connections.
_pending_session_tasks: dict[str, asyncio.Task[DeepgramSession]] = {}

# Guards all mutations to both dicts above.
# NOTE:
# asyncio.Lock() at module import time is acceptable in Python 3.10+ because
# lock construction no longer requires a running event loop. In production this
# module is expected to run on a single application event loop, so this is low
# risk. In tests that repeatedly create independent loops (for example via
# asyncio.run()), be aware that module-level asyncio primitives can end up bound
# to a different loop than the one currently executing. If that ever becomes an
# issue, prefer initializing such locks during app startup under the active loop.
_sessions_lock = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _close_connection_safely(
    connection: DeepgramConnection,
    *,
    session_id: str,
) -> None:
    """Best-effort close: always returns, never raises."""
    try:
        await asyncio.wait_for(connection.close(), timeout=CLOSE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error(
            "Timed out while closing Deepgram connection for session %s",
            session_id,
        )
    except Exception:
        logger.exception(
            "Unexpected error while closing Deepgram connection for session %s",
            session_id,
        )


async def _cancel_task_safely(task: asyncio.Task[Any], *, label: str) -> None:
    """Cancel a task and suppress the expected shutdown exceptions."""
    if task.done():
        # Drain any stored exception so it isn't logged as unhandled.
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
        return

    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        logger.debug("Cancelled task: %s", label)
    except Exception:
        logger.exception("Task raised during cancellation: %s", label)


async def _teardown_session_resources(
    session: DeepgramSession,
    *,
    close_connection: bool,
) -> None:
    """Cancel the keepalive task and optionally close the Deepgram WebSocket.

    Args:
        session: The session whose resources should be torn down.
        close_connection: Pass True when the underlying WebSocket should be
            gracefully closed (normal session end). Pass False when the
            WebSocket is already known dead — e.g. after DeepgramConnectionError
            — to skip the redundant close attempt.
    """
    if session.keepalive_task is not None:
        await _cancel_task_safely(
            session.keepalive_task,
            label=f"keepalive:{session.session_id}",
        )
        session.keepalive_task = None

    if close_connection:
        await _close_connection_safely(
            session.connection,
            session_id=session.session_id,
        )


async def _remove_active_session_if_same(
    session_id: str,
    expected_session: DeepgramSession,
) -> bool:
    """Atomically remove a session only if it still matches expected_session.

    Prevents a race where session A is torn down just as session B for the
    same session_id is being created — we must not remove session B's entry.

    Returns:
        True if the entry was removed, False if it had already been replaced.
    """
    async with _sessions_lock:
        current = _active_sessions.get(session_id)
        if current is expected_session:
            _active_sessions.pop(session_id, None)
            return True
        return False


async def _keepalive_loop(session: DeepgramSession) -> None:
    """Send Deepgram KeepAlive messages while the session is silent.

    Deepgram closes idle WebSocket connections after ~10s without receiving
    audio or a keepalive. This loop wakes up every KEEPALIVE_INTERVAL_SECONDS,
    checks how long it has been since the last audio frame, and sends a
    keepalive if the session has been quiet for KEEPALIVE_IDLE_THRESHOLD_SECONDS.

    Exits cleanly on task cancellation (session close) or if the connection
    is permanently lost.
    """
    session_id = session.session_id
    try:
        while True:
            await asyncio.sleep(KEEPALIVE_INTERVAL_SECONDS)

            if not session.connection.is_connected:
                logger.info(
                    "Keepalive loop exiting for session %s — connection no longer active",
                    session_id,
                )
                return

            idle_for = time.monotonic() - session.last_audio_monotonic
            if idle_for < KEEPALIVE_IDLE_THRESHOLD_SECONDS:
                # Audio is flowing — Deepgram already knows we're alive.
                continue

            try:
                await asyncio.wait_for(
                    session.connection.send_keepalive(),
                    timeout=SEND_TIMEOUT_SECONDS,
                )
                logger.debug(
                    "Sent KeepAlive for session %s after %.2fs idle",
                    session_id,
                    idle_for,
                )

            except asyncio.TimeoutError:
                logger.warning(
                    "Keepalive timed out for session %s — socket may be hanging",
                    session_id,
                )
                # Don't exit — give Deepgram's own reconnect logic a chance
                # to recover on the next cycle.

            except DeepgramConnectionError as exc:
                logger.warning(
                    "Keepalive failed for session %s: %s — exiting keepalive loop",
                    session_id,
                    exc,
                )
                return

            except Exception:
                logger.exception(
                    "Unexpected error in keepalive loop for session %s — exiting",
                    session_id,
                )
                return

    except asyncio.CancelledError:
        logger.debug("Keepalive loop cancelled for session %s", session_id)
        raise  # Must re-raise so asyncio marks the task as cancelled correctly.


async def _build_session(
    session_id: str,
    on_transcript: Callable[[TranscriptResult], Coroutine[Any, Any, None]],
    on_utterance_end: Callable[[], Coroutine[Any, Any, None]] | None,
) -> DeepgramSession:
    """Create a DeepgramConnection, connect it, and start the keepalive loop.

    This is the only place a DeepgramSession is constructed. It is always
    called as an asyncio.Task so concurrent callers share one attempt.

    Raises:
        DeepgramConnectionError: If the Deepgram WebSocket handshake fails.
        asyncio.TimeoutError: If the handshake does not complete within
            CONNECT_TIMEOUT_SECONDS.
    """
    connection = DeepgramConnection(
        on_transcript=on_transcript,
        on_utterance_end=on_utterance_end,
    )

    # connect() does network I/O. We wrap it in a timeout so a slow/unreachable
    # Deepgram endpoint does not hang the session startup indefinitely.
    await asyncio.wait_for(connection.connect(), timeout=CONNECT_TIMEOUT_SECONDS)

    session = DeepgramSession(
        session_id=session_id,
        connection=connection,
    )
    session.keepalive_task = asyncio.create_task(
        _keepalive_loop(session),
        name=f"deepgram-keepalive:{session_id}",
    )
    return session


# ─────────────────────────────────────────────────────────────────────────────
# Public lifecycle API  (called by orchestrator / FastAPI WS handler)
# ─────────────────────────────────────────────────────────────────────────────


async def create_deepgram_connection(
    session_id: str,
    on_transcript: Callable[[TranscriptResult], Coroutine[Any, Any, None]],
    on_utterance_end: Callable[[], Coroutine[Any, Any, None]] | None = None,
) -> DeepgramConnection:
    """Create and register a Deepgram session for a BEAN session.

    Safe against concurrent duplicate calls: if two callers race on the same
    session_id, only one WebSocket connection is created and both callers
    receive the same DeepgramConnection object.

    Args:
        session_id: Unique BEAN session identifier. Must not be empty.
        on_transcript: Async callback fired for every Deepgram transcript
            result (partial and final). Receives a TranscriptResult.
        on_utterance_end: Optional async callback fired when Deepgram signals
            the end of a speech segment. The orchestrator uses this to trigger
            response generation.

    Returns:
        The live DeepgramConnection for this session.

    Raises:
        ValueError: If session_id is empty.
        DeepgramConnectionError: If the Deepgram handshake fails.
        asyncio.TimeoutError: If the handshake times out.
    """
    if not session_id:
        raise ValueError("session_id must not be empty")

    async with _sessions_lock:
        # Fast path: a healthy connection already exists — reuse it.
        # This handles the case where the orchestrator calls create twice,
        # e.g. during a session resume or a test retry.
        existing = _active_sessions.get(session_id)
        if existing is not None and existing.is_connected:
            logger.debug("Reusing existing Deepgram session for session %s", session_id)
            return existing.connection

        # If a stale disconnected session object is present, discard it so
        # _build_session starts fresh instead of wrapping a dead socket.
        if existing is not None:
            logger.warning(
                "Stale Deepgram session found for session %s — discarding before reconnect",
                session_id,
            )
            _active_sessions.pop(session_id, None)

        # Deduplication: if another caller already launched a connect task for
        # this session_id, await that same task instead of launching a second one.
        pending = _pending_session_tasks.get(session_id)
        if pending is None:
            pending = asyncio.create_task(
                _build_session(session_id, on_transcript, on_utterance_end),
                name=f"deepgram-connect:{session_id}",
            )
            _pending_session_tasks[session_id] = pending
            logger.info("Started Deepgram session creation for session %s", session_id)
        else:
            logger.info(
                "Awaiting in-flight Deepgram session creation for session %s",
                session_id,
            )

    # Await outside the lock: connect() does network I/O and may take several
    # seconds. Holding the lock here would block all other sessions.
    try:
        session = await pending
    except Exception as exc:
        # Clean up the pending entry so the next attempt can start fresh.
        async with _sessions_lock:
            if _pending_session_tasks.get(session_id) is pending:
                _pending_session_tasks.pop(session_id, None)
        logger.exception(
            "Deepgram session creation failed for session %s: %s",
            session_id,
            exc,
        )
        raise

    # Register the new session. Re-check under the lock for the duplicate
    # window: two tasks could have been racing if the lock logic above had
    # a brief gap between checking and creating the pending task.
    async with _sessions_lock:
        if _pending_session_tasks.get(session_id) is pending:
            _pending_session_tasks.pop(session_id, None)

        current_active = _active_sessions.get(session_id)
        is_duplicate = (
            current_active is not None
            and current_active is not session
            and current_active.is_connected
        )

        if not is_duplicate:
            _active_sessions[session_id] = session

    if is_duplicate:
        # Another task won the race. Tear down the duplicate we just built
        # and return the winner's connection instead.
        logger.warning(
            "Duplicate Deepgram session created for session %s — "
            "closing duplicate and reusing existing",
            session_id,
        )
        await _teardown_session_resources(session, close_connection=True)

        winner = _active_sessions.get(session_id)
        if winner is None:
            # Extremely unlikely: the winner was closed between the lock
            # release above and this read.
            raise DeepgramConnectionError(
                f"Deepgram duplicate-session resolution failed for session {session_id}: "
                "no active session found after race"
            )
        return winner.connection

    logger.info(
        "Deepgram session ready for session %s (active sessions: %d)",
        session_id,
        len(_active_sessions),
    )
    return session.connection


async def close_deepgram_connection(session_id: str) -> None:
    """Close and deregister the Deepgram session for a BEAN session.

    Safe to call multiple times and safe to call when no session exists.
    Also cancels any in-flight connect attempt for the session_id.

    Args:
        session_id: Session to close.
    """
    if not session_id:
        logger.warning(
            "close_deepgram_connection called with empty session_id — ignoring"
        )
        return

    async with _sessions_lock:
        session = _active_sessions.pop(session_id, None)
        pending = _pending_session_tasks.pop(session_id, None)

    # Cancel any in-flight connect attempt first so it doesn't register
    # a new session after we just removed the entry.
    if pending is not None:
        await _cancel_task_safely(
            pending,
            label=f"pending-connect:{session_id}",
        )

    if session is None:
        logger.info(
            "close_deepgram_connection: no active session for %s — nothing to close",
            session_id,
        )
        return

    await _teardown_session_resources(session, close_connection=True)

    logger.info(
        "Deepgram session closed for session %s (active sessions: %d)",
        session_id,
        len(_active_sessions),
    )


def get_deepgram_connection(session_id: str) -> DeepgramConnection | None:
    """Return the live DeepgramConnection for a session, or None.

    Does not acquire the lock — safe to call from the hot audio path.
    """
    session = _active_sessions.get(session_id)
    return session.connection if session is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# Production hot path — raw bytes from FastAPI WebSocket binary frames
# ─────────────────────────────────────────────────────────────────────────────


async def stream_audio_chunk(
    session_id: str,
    audio_chunk: bytes,
) -> dict[str, Any]:
    """Forward a raw PCM audio chunk to Deepgram.

    This is the production hot path. The FastAPI WebSocket handler calls
    this directly for every binary frame received from the ESP32.
    Do NOT use deepgram_transcribe() for live robot audio — the base64
    encode/decode adds latency and CPU overhead on every frame.

    Args:
        session_id: BEAN session identifier.
        audio_chunk: Raw PCM16 mono 16kHz bytes from the robot.

    Returns:
        Status dict. Shapes:

        {"status": "audio_sent",     "session_id": str, "bytes": int}
        {"status": "connecting",     "session_id": str, "message": str}
        {"status": "no_connection",  "session_id": str, "message": str}
        {"status": "disconnected",   "session_id": str, "message": str}
        {"status": "connection_lost","session_id": str, "message": str}
        {"status": "invalid_input",  "session_id": str, "message": str}
        {"status": "error",          "session_id": str, "error": str}
    """
    # ── Input validation ───────────────────────────────────────────────────
    if not session_id:
        logger.warning(
            "stream_audio_chunk called with empty session_id — dropping frame"
        )
        return {
            "status": "invalid_input",
            "session_id": session_id,
            "message": "session_id must not be empty",
        }

    if not audio_chunk:
        logger.debug(
            "stream_audio_chunk got empty audio_chunk for session %s — dropping",
            session_id,
        )
        return {
            "status": "invalid_input",
            "session_id": session_id,
            "message": "audio_chunk must not be empty",
        }

    # ── Session lookup (no lock — reads are safe on CPython) ──────────────
    session = _active_sessions.get(session_id)

    if session is None:
        # Distinguish "never started" from "still connecting" so the WS
        # handler can log the right warning and decide whether to retry.
        if _pending_session_tasks.get(session_id) is not None:
            return {
                "status": "connecting",
                "session_id": session_id,
                "message": "Deepgram session is still being established — drop this frame and retry.",
            }
        return {
            "status": "no_connection",
            "session_id": session_id,
            "message": "No Deepgram session exists. Orchestrator must call create_deepgram_connection() first.",
        }

    if not session.connection.is_connected:
        # The session exists but the WebSocket has dropped. Deepgram's own
        # reconnect logic (max 3 attempts) may still be running inside
        # deepgram_service.py. Rate-limit the log so we don't flood on
        # every frame at 50fps.
        now = time.monotonic()
        if (
            now - session.last_disconnected_log_monotonic
            >= DISCONNECTED_LOG_INTERVAL_SECONDS
        ):
            logger.info(
                "Deepgram connection inactive for session %s — "
                "reconnect may be in progress (suppressing further logs for %.0fs)",
                session_id,
                DISCONNECTED_LOG_INTERVAL_SECONDS,
            )
            session.last_disconnected_log_monotonic = now
        return {
            "status": "disconnected",
            "session_id": session_id,
            "message": "Deepgram WebSocket is not currently connected. Reconnect may be in progress.",
        }

    # ── Audio forwarding ───────────────────────────────────────────────────
    session.last_audio_monotonic = time.monotonic()

    try:
        await asyncio.wait_for(
            session.connection.send_audio(audio_chunk),
            timeout=SEND_TIMEOUT_SECONDS,
        )
        return {
            "status": "audio_sent",
            "session_id": session_id,
            "bytes": len(audio_chunk),
        }

    except asyncio.TimeoutError:
        logger.error(
            "Timed out sending audio chunk to Deepgram for session %s "
            "(chunk size: %d bytes)",
            session_id,
            len(audio_chunk),
        )
        return {
            "status": "error",
            "session_id": session_id,
            "error": "Timed out sending audio chunk to Deepgram",
        }

    except DeepgramConnectionError as exc:
        # Deepgram's internal reconnect has already been exhausted — the
        # connection is permanently dead. Remove it from the registry so
        # the next frame gets "no_connection" (a clear signal to the
        # orchestrator to act) rather than silently looping on "disconnected".
        logger.error(
            "Deepgram connection permanently lost for session %s: %s",
            session_id,
            exc,
        )
        removed = await _remove_active_session_if_same(session_id, session)
        if removed:
            await _teardown_session_resources(session, close_connection=False)

        return {
            "status": "connection_lost",
            "session_id": session_id,
            "message": f"Deepgram connection permanently lost: {exc}",
        }

    except Exception:
        logger.exception(
            "Unexpected error while streaming audio for session %s",
            session_id,
        )
        return {
            "status": "error",
            "session_id": session_id,
            "error": "Unexpected error while forwarding audio to Deepgram",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Fallback ADK path — base64 input for testing / non-hot-path orchestration
# ─────────────────────────────────────────────────────────────────────────────


async def deepgram_transcribe(
    session_id: str,
    audio_b64: str,
) -> dict[str, Any]:
    """Decode a base64 audio payload and forward it to Deepgram.

    This is a thin wrapper around stream_audio_chunk() for use as an ADK
    FunctionTool. It should NOT be used for the live robot audio firehose —
    use stream_audio_chunk() directly from the FastAPI WebSocket handler
    to avoid the base64 decode overhead on every frame.

    Args:
        session_id: BEAN session identifier.
        audio_b64: Base64-encoded PCM16 mono 16kHz audio.

    Returns:
        Same status dict shapes as stream_audio_chunk(), plus:
        {"status": "invalid_input", ..., "message": "audio_b64 is not valid base64"}
    """
    if not session_id:
        logger.warning(
            "deepgram_transcribe called with empty session_id — dropping frame"
        )
        return {
            "status": "invalid_input",
            "session_id": session_id,
            "message": "session_id must not be empty",
        }

    if not audio_b64:
        logger.debug(
            "deepgram_transcribe called with empty audio_b64 for session %s", session_id
        )
        return {
            "status": "invalid_input",
            "session_id": session_id,
            "message": "audio_b64 must not be empty",
        }

    try:
        # validate=True rejects characters outside the base64 alphabet, catching
        # corrupted payloads before they reach the WebSocket send path.
        audio_chunk = base64.b64decode(audio_b64, validate=True)
    except Exception as exc:
        logger.warning(
            "Invalid base64 audio payload for session %s: %s",
            session_id,
            exc,
        )
        return {
            "status": "invalid_input",
            "session_id": session_id,
            "message": "audio_b64 is not valid base64",
        }

    return await stream_audio_chunk(session_id, audio_chunk)


# ─────────────────────────────────────────────────────────────────────────────
# ADK registration
# ─────────────────────────────────────────────────────────────────────────────

deepgram_transcribe_tool = FunctionTool(func=deepgram_transcribe)