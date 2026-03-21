"""BEAN AI v1 — Deepgram WebSocket STT service.

Manages a persistent Deepgram WebSocket connection for one BEAN session.
Audio bytes flow: ESP32 → Cloud Run RAM → Deepgram → transcript text.
Audio is NEVER written to disk or any storage at any point.


"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any

import websockets
from websockets.exceptions import ConnectionClosed

from shared.config import get_settings
from shared.exceptions import DeepgramConnectionError
from shared.schemas import TranscriptResult

logger = logging.getLogger(__name__)


class DeepgramConnection:
    """Manages a persistent Deepgram WebSocket for one BEAN user session.

    Lifecycle:
        connect() → send_audio() / send_audio_b64() [many times] → close()

    Privacy:
        Audio bytes are passed in from the WebSocket handler and forwarded
        to Deepgram. They are never stored — only the resulting transcript
        text is returned via the on_transcript callback.

    Thread safety:
        All methods are async and intended to be called from a single
        asyncio event loop. Do not share one instance across event loops.
    """

    MAX_RECONNECT_ATTEMPTS = 3
    RECONNECT_DELAY_SECONDS = 2.0

    def __init__(
        self,
        on_transcript: Callable[[TranscriptResult], Coroutine[Any, Any, None]],
        on_utterance_end: Callable[[], Coroutine[Any, Any, None]] | None = None,
    ) -> None:
        self._settings = get_settings()
        self._ws: Any | None = None
        self._on_transcript = on_transcript
        self._on_utterance_end = on_utterance_end
        self._receive_task: asyncio.Task[None] | None = None
        self._connected = False
        self._closing = False
        self._reconnect_attempts = 0

    # ─────────────────────────────────────────────────────────────────────────
    # Public properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """True only if the WebSocket is open AND not in the process of closing.

        Uses getattr for the closed-state check because websockets >= 13 replaced
        the .closed bool attribute with .close_code (None = open, int = closed).
        getattr handles both versions safely.
        """
        return (
            self._connected
            and not self._closing
            and self._ws is not None
            and getattr(self._ws, "close_code", None) is None
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _build_url(self) -> str:
        """Build the Deepgram streaming WebSocket URL from settings."""
        s = self._settings
        params = {
            "model": s.deepgram_model,
            "language": s.deepgram_language,
            "sample_rate": str(s.deepgram_sample_rate),
            "encoding": s.deepgram_encoding,
            "channels": str(s.deepgram_channels),
            "endpointing": str(s.deepgram_endpointing_ms),
            "utterance_end_ms": str(s.deepgram_utterance_end_ms),
            "interim_results": "true",
            "punctuate": "true",
            "smart_format": "true",
        }
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"wss://api.deepgram.com/v1/listen?{query}"

    # ─────────────────────────────────────────────────────────────────────────
    # Connection lifecycle
    # ─────────────────────────────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the Deepgram WebSocket connection.

        Idempotent — safe to call if already connected (returns immediately).

        Raises:
            DeepgramConnectionError: If the WebSocket handshake fails for any
                reason (bad API key, network unreachable, Deepgram down, etc.).
        """
        if self._connected:
            logger.debug("Deepgram already connected — skipping connect()")
            return

        try:
            url = self._build_url()
            headers = {"Authorization": f"Token {self._settings.deepgram_api_key}"}

            self._ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=self._settings.ws_ping_interval_seconds,
                ping_timeout=10,
                close_timeout=5,
                max_size=2**20,  # 1MB max inbound message size
            )
            self._connected = True
            self._closing = False
            self._reconnect_attempts = 0

            # Start the receive loop as a background task so transcript
            # results arrive asynchronously while the caller continues.
            self._receive_task = asyncio.create_task(
                self._receive_loop(),
                name="deepgram-receive",
            )
            logger.info("Deepgram WebSocket connected")

        except Exception as exc:
            logger.error("Deepgram connection failed: %s", exc)
            self._connected = False
            raise DeepgramConnectionError(str(exc)) from exc

    async def close(self) -> None:
        """Gracefully close the Deepgram connection.

        Sends Deepgram's CloseStream signal, cancels the receive loop, then
        closes the underlying WebSocket. Safe to call even if not connected
        and safe to call multiple times — subsequent calls are no-ops.
        """
        if self._closing:
            return

        self._closing = True
        self._connected = False

        # Cancel the background receive loop first so it stops processing
        # messages while we are tearing down.
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await asyncio.wait_for(self._receive_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        # Send the Deepgram close signal and close the underlying WebSocket.
        if self._ws and getattr(self._ws, "close_code", None) is None:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                await asyncio.wait_for(self._ws.close(), timeout=3.0)
            except Exception as exc:
                # Errors during close are non-critical — the connection is
                # going away regardless. Log at debug so production logs are
                # not flooded on normal session teardown.
                logger.debug("Deepgram close error (non-critical): %s", exc)

        self._ws = None
        logger.info("Deepgram WebSocket closed")

    # ─────────────────────────────────────────────────────────────────────────
    # Audio sending
    # ─────────────────────────────────────────────────────────────────────────

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Forward raw PCM16 audio bytes to Deepgram.

        This is the production hot path — called by stream_audio_chunk() in
        the STT agent on every binary frame from the ESP32. Keep it lean.

        Privacy: audio_bytes are forwarded to Deepgram over the live WebSocket
        and are never written to disk, logged, or stored anywhere.

        Args:
            audio_bytes: Raw PCM16 mono 16kHz bytes. Empty input is silently
                ignored — the caller should filter these upstream.
        """
        if not audio_bytes:
            logger.debug("send_audio called with empty bytes — skipping")
            return

        if not self.is_connected or self._ws is None:
            logger.debug("send_audio skipped — Deepgram not connected")
            return

        try:
            await self._ws.send(audio_bytes)

        except ConnectionClosed:
            logger.warning(
                "Deepgram connection closed during send_audio — attempting reconnect"
            )
            self._connected = False
            await self._try_reconnect()

        except Exception as exc:
            logger.error("send_audio unexpected error: %s", exc)
            self._connected = False

    async def send_audio_b64(self, audio_b64: str) -> None:
        """Decode a base64 audio payload and forward it to Deepgram.

        Convenience wrapper around send_audio() for callers that receive
        audio as base64-encoded strings — primarily the ADK FunctionTool
        fallback path in the STT agent and Dev B's WebSocket handler.

        The production hot path (stream_audio_chunk in stt/agent.py) calls
        send_audio() directly to avoid the decode overhead on every frame.

        Args:
            audio_b64: Base64-encoded PCM16 mono 16kHz audio.

        Raises:
            ValueError: If audio_b64 is not valid base64. The caller should
                return an invalid_input status and drop the frame.
            DeepgramConnectionError: Propagated from send_audio() if the
                connection is permanently lost.
        """
        if not audio_b64:
            logger.debug("send_audio_b64 called with empty string — skipping")
            return

        try:
            # validate=True rejects characters outside the base64 alphabet,
            # catching corrupted payloads before they reach the WebSocket.
            audio_bytes = base64.b64decode(audio_b64, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 audio payload: {exc}") from exc

        await self.send_audio(audio_bytes)

    async def send_keepalive(self) -> None:
        """Send a Deepgram KeepAlive message to prevent idle connection timeout.

        Deepgram closes WebSocket connections that receive no audio or keepalive
        signal for approximately 10 seconds. The STT agent's keepalive loop
        calls this method every ~4 seconds when the session has been silent for
        more than 3.5 seconds.

        Why a separate method instead of send_audio(b"")?
            Deepgram treats an empty binary frame differently from a JSON
            KeepAlive — the former may be interpreted as end-of-stream on
            some model configurations. The explicit JSON message is the
            documented approach in the Deepgram Nova-2 API.

        Raises:
            DeepgramConnectionError: If the connection is not open or the send
                fails. The STT agent's keepalive loop catches this and exits,
                allowing Deepgram's own reconnect logic to run.
        """
        if not self.is_connected or self._ws is None:
            raise DeepgramConnectionError(
                "Cannot send keepalive — Deepgram WebSocket is not connected"
            )

        try:
            await self._ws.send('{"type": "KeepAlive"}')

        except ConnectionClosed as exc:
            logger.warning("Deepgram connection closed during send_keepalive")
            self._connected = False
            raise DeepgramConnectionError(
                "Deepgram connection closed during keepalive send"
            ) from exc

        except Exception as exc:
            logger.error("send_keepalive unexpected error: %s", exc)
            self._connected = False
            raise DeepgramConnectionError(f"Keepalive send failed: {exc}") from exc

    # ─────────────────────────────────────────────────────────────────────────
    # Receive loop
    # ─────────────────────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        """Continuously receive and dispatch messages from Deepgram.

        Runs as a background asyncio task for the lifetime of the connection.
        Handles four Deepgram message types:
            Results       — transcript (partial or final), fires on_transcript
            UtteranceEnd  — end of speech segment, fires on_utterance_end
            Metadata      — logged at debug, otherwise ignored
            Error         — logged at error level
        """
        if self._ws is None:
            return

        try:
            async for raw_message in self._ws:
                if self._closing:
                    break

                try:
                    data = json.loads(raw_message)
                    msg_type = data.get("type", "")

                    if msg_type == "Results":
                        await self._handle_results(data)

                    elif msg_type == "UtteranceEnd":
                        # Deepgram signals the user has stopped speaking.
                        # The orchestrator uses this to trigger response generation.
                        if self._on_utterance_end is not None:
                            await self._on_utterance_end()

                    elif msg_type == "Metadata":
                        logger.debug("Deepgram metadata received: %s", data)

                    elif msg_type == "Error":
                        logger.error(
                            "Deepgram sent error message: code=%s description=%s",
                            data.get("error_code", "unknown"),
                            data.get("description", "no description"),
                        )

                    else:
                        logger.debug("Deepgram unrecognised message type: %s", msg_type)

                except json.JSONDecodeError:
                    logger.warning(
                        "Deepgram sent a non-JSON message — ignoring (len=%d)",
                        len(raw_message)
                        if isinstance(raw_message, (str, bytes))
                        else -1,
                    )
                except Exception:
                    logger.exception("Unhandled error processing Deepgram message")

        except asyncio.CancelledError:
            # Normal shutdown — task was cancelled by close().
            logger.debug("Deepgram receive loop cancelled")

        except ConnectionClosed as exc:
            logger.warning("Deepgram connection closed in receive loop: %s", exc)
            if not self._closing:
                self._connected = False
                await self._try_reconnect()

        except Exception:
            logger.exception("Deepgram receive loop encountered an unexpected error")
            self._connected = False

    async def _handle_results(self, data: dict[str, Any]) -> None:
        """Parse a Deepgram Results message and fire the transcript callback.

        Deepgram sends Results for both interim (partial) and final transcripts.
        Both trigger the callback so the orchestrator can show partial results
        while the final result is still being processed.

        Silently drops empty transcripts — these are common at utterance
        boundaries and at connection start before the model has warmed up.
        """
        try:
            channel = data.get("channel", {})
            alternatives = channel.get("alternatives", [])
            if not alternatives:
                return

            best = alternatives[0]
            text = best.get("transcript", "").strip()
            if not text:
                return

            confidence = float(best.get("confidence", 0.0))
            is_final = data.get("is_final", False)
            speech_final = data.get("speech_final", False)

            result = TranscriptResult(
                type="transcript_final"
                if (is_final or speech_final)
                else "transcript_partial",
                text=text,
                confidence=confidence,
                is_final=is_final or speech_final,
            )
            await self._on_transcript(result)

        except Exception:
            logger.exception("Error parsing Deepgram Results message")

    # ─────────────────────────────────────────────────────────────────────────
    # Reconnection
    # ─────────────────────────────────────────────────────────────────────────

    async def _try_reconnect(self) -> None:
        """Attempt to reconnect to Deepgram after a connection drop.

        Uses linear back-off (2s × attempt number) and gives up after
        MAX_RECONNECT_ATTEMPTS. This is intentionally conservative — if
        Deepgram is down, we do not want Cloud Run burning CPU in a tight
        retry loop.

        Does nothing if close() has been called (_closing is True), because
        the connection drop is intentional in that case.
        """
        if self._closing:
            return

        if self._reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
            logger.error(
                "Deepgram reconnect giving up after %d failed attempts",
                self.MAX_RECONNECT_ATTEMPTS,
            )
            return

        self._reconnect_attempts += 1
        delay = self.RECONNECT_DELAY_SECONDS * self._reconnect_attempts

        logger.info(
            "Deepgram reconnect attempt %d/%d in %.1fs",
            self._reconnect_attempts,
            self.MAX_RECONNECT_ATTEMPTS,
            delay,
        )

        await asyncio.sleep(delay)

        try:
            await self.connect()
            logger.info(
                "Deepgram reconnected successfully on attempt %d",
                self._reconnect_attempts,
            )
        except DeepgramConnectionError as exc:
            logger.warning(
                "Deepgram reconnect attempt %d failed: %s",
                self._reconnect_attempts,
                exc,
            )
