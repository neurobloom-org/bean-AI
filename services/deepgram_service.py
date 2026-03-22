"""BEAN AI — Deepgram WebSocket STT service.

Manages a persistent Deepgram WebSocket connection for one BEAN session.
Audio flows: ESP32 → Cloud Run RAM → Deepgram → transcript callbacks.
Audio is NEVER written to disk or any storage layer at any point.

Implementation note:
    This service uses the `websockets` library directly rather than the
    Deepgram Python SDK. The SDK is not listed as a dependency. The hand-
    rolled implementation gives us full control over reconnect timing,
    keepalive framing, and the receive loop — all of which need to behave
    precisely for real-time robot audio.

Exported:
    DeepgramConnection   — manages one session's WebSocket lifecycle
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import Callable, Coroutine
from typing import Any, Final

import websockets
from websockets.exceptions import ConnectionClosed

from shared.config import get_settings
from shared.exceptions import DeepgramConnectionError
from shared.schemas import TranscriptResult

__all__ = ["DeepgramConnection"]

logger = logging.getLogger(__name__)

# ── Module-level constants ────────────────────────────────────────────────────

#: How many times to attempt reconnect before giving up permanently.
MAX_RECONNECT_ATTEMPTS: Final[int] = 3

#: Base delay between reconnect attempts. Multiplied by attempt number
#: for linear back-off: 2s, 4s, 6s.
RECONNECT_DELAY_SECONDS: Final[float] = 2.0

#: Timeout for the WebSocket close() call during teardown.
_CLOSE_TIMEOUT_SECONDS: Final[float] = 3.0

#: Timeout for cancelling the receive task during teardown.
_RECEIVE_TASK_CANCEL_TIMEOUT_SECONDS: Final[float] = 2.0

#: 1 MB max inbound message size — Deepgram sends JSON, never large payloads.
_MAX_WS_MESSAGE_SIZE: Final[int] = 2**20


# ── DeepgramConnection ────────────────────────────────────────────────────────


class DeepgramConnection:
    """Manages a persistent Deepgram WebSocket for one BEAN user session.

    Lifecycle:
        connect() → send_audio() / send_keepalive() [many times] → close()

    Thread safety:
        All methods are async and must be called from a single asyncio event
        loop. Do not share one instance across event loops or threads.

    Reconnection:
        On unexpected connection loss, _try_reconnect() makes up to
        MAX_RECONNECT_ATTEMPTS attempts with linear back-off. If all attempts
        fail, is_connected returns False and the STT agent removes the session
        from its registry so the WS handler can restart cleanly.

    Privacy:
        Audio bytes are forwarded to Deepgram in RAM only. They are never
        written to disk, Supabase, or any other storage layer.
    """

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

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        """True only if the WebSocket is open and not in the process of closing.

        Uses getattr for the closed-state check because websockets >= 13
        replaced the .closed bool attribute with .close_code (None = open,
        int = closed). getattr handles both versions safely.
        """
        return (
            self._connected
            and not self._closing
            and self._ws is not None
            and getattr(self._ws, "close_code", None) is None
        )

    # ── URL builder ───────────────────────────────────────────────────────────

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

    # ── Connection lifecycle ──────────────────────────────────────────────────

    async def connect(self) -> None:
        """Open the Deepgram WebSocket connection and start the receive loop.

        Idempotent — safe to call if already connected (returns immediately).

        Raises:
            DeepgramConnectionError: If DEEPGRAM_API_KEY is not configured,
                or if the WebSocket handshake fails for any reason (bad key,
                network unreachable, Deepgram down, etc.).
        """
        if self._connected:
            logger.debug("Deepgram already connected — skipping connect()")
            return

        # Validate configuration upfront — a missing key produces a confusing
        # 401 HTTP error from Deepgram otherwise.
        if not self._settings.deepgram_api_key:
            raise DeepgramConnectionError(
                "DEEPGRAM_API_KEY is not configured — cannot connect to Deepgram"
            )

        try:
            url = self._build_url()
            headers = {"Authorization": f"Token {self._settings.deepgram_api_key}"}

            self._ws = await websockets.connect(
                url,
                additional_headers=headers,
                ping_interval=self._settings.ws_ping_interval_seconds,
                ping_timeout=10,
                close_timeout=5,
                max_size=_MAX_WS_MESSAGE_SIZE,
            )
            self._connected = True
            self._closing = False
            self._reconnect_attempts = 0

            self._receive_task = asyncio.create_task(
                self._receive_loop(),
                name="deepgram-receive",
            )
            logger.info("Deepgram WebSocket connected")

        except DeepgramConnectionError:
            raise
        except Exception as exc:
            self._connected = False
            logger.error("Deepgram connection failed: %s", exc)
            raise DeepgramConnectionError(
                f"Deepgram WebSocket handshake failed: {exc}"
            ) from exc

    async def close(self) -> None:
        """Gracefully close the Deepgram connection.

        Sends Deepgram's CloseStream signal, cancels the receive loop, then
        closes the underlying WebSocket. Safe to call even if not connected,
        and safe to call multiple times — subsequent calls are no-ops.
        """
        if self._closing:
            return

        self._closing = True
        self._connected = False

        # Cancel the receive loop before closing the socket so it doesn't
        # try to process messages while we're tearing down.
        await self._cancel_receive_task()

        # Send the Deepgram CloseStream signal then close the socket.
        if self._ws is not None and getattr(self._ws, "close_code", None) is None:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                await asyncio.wait_for(
                    self._ws.close(), timeout=_CLOSE_TIMEOUT_SECONDS
                )
            except Exception as exc:
                # Non-critical — the connection is going away regardless.
                logger.debug("Deepgram close error (non-critical): %s", exc)

        self._ws = None
        logger.info("Deepgram WebSocket closed")

    async def _cancel_receive_task(self) -> None:
        """Cancel and await the receive task, consuming any stored exception."""
        if self._receive_task is None or self._receive_task.done():
            if self._receive_task is not None and self._receive_task.done():
                # Drain any stored exception so it isn't reported as unhandled.
                try:
                    self._receive_task.result()
                except (asyncio.CancelledError, Exception):
                    pass
            return

        self._receive_task.cancel()
        try:
            await asyncio.wait_for(
                self._receive_task,
                timeout=_RECEIVE_TASK_CANCEL_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            pass
        except asyncio.TimeoutError:
            logger.warning(
                "Deepgram receive task did not cancel within %.1fs",
                _RECEIVE_TASK_CANCEL_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            # Task raised an unexpected exception before it was cancelled.
            # Log it so it's visible but don't let it block teardown.
            logger.error(
                "Deepgram receive task raised on cancel: %s", exc
            )

    # ── Audio sending ─────────────────────────────────────────────────────────

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Forward raw PCM16 audio bytes to Deepgram.

        This is the production hot path — called by stream_audio_chunk() in
        the STT agent on every binary frame from the ESP32. Keep it lean.

        Privacy: audio_bytes are forwarded over the live WebSocket only.
        They are never written to disk, logged, or stored anywhere.

        Args:
            audio_bytes: Raw PCM16 mono 16kHz bytes. Empty bytes are silently
                ignored — the caller should filter these upstream.

        Raises:
            DeepgramConnectionError: If the send fails due to a permanent
                connection loss (after reconnect attempts are exhausted).
                The STT agent catches this and removes the session.
        """
        if not audio_bytes:
            logger.debug("send_audio: empty bytes — skipping")
            return

        if not self.is_connected or self._ws is None:
            logger.debug("send_audio: not connected — dropping frame")
            return

        try:
            await self._ws.send(audio_bytes)

        except ConnectionClosed:
            logger.warning(
                "Deepgram connection closed during send_audio — attempting reconnect"
            )
            self._connected = False
            await self._try_reconnect()
            # After reconnect, the frame that triggered this is dropped.
            # The next frame from the ESP32 will flow through the new connection.

        except Exception as exc:
            # Unexpected error (e.g. websockets internal error, OS error).
            # Mark disconnected and raise so the STT agent can act.
            logger.error("send_audio unexpected error: %s — marking disconnected", exc)
            self._connected = False
            raise DeepgramConnectionError(
                f"Deepgram send_audio failed: {exc}"
            ) from exc

    async def send_audio_b64(self, audio_b64: str) -> None:
        """Decode a base64 audio payload and forward it to Deepgram.

        Convenience wrapper around send_audio() for callers that receive audio
        as base64 strings — the ADK FunctionTool fallback path in the STT agent.

        The production hot path calls send_audio() directly to avoid the
        base64 decode overhead on every 50fps frame.

        Args:
            audio_b64: Base64-encoded PCM16 mono 16kHz audio.

        Raises:
            ValueError:              If audio_b64 is not valid base64.
            DeepgramConnectionError: From send_audio() on permanent loss.
        """
        if not audio_b64:
            logger.debug("send_audio_b64: empty string — skipping")
            return

        try:
            audio_bytes = base64.b64decode(audio_b64, validate=True)
        except Exception as exc:
            raise ValueError(f"Invalid base64 audio payload: {exc}") from exc

        await self.send_audio(audio_bytes)

    async def send_keepalive(self) -> None:
        """Send a Deepgram KeepAlive message to prevent idle timeout.

        Deepgram closes WebSocket connections that receive no audio or keepalive
        signal for ~10 seconds. The STT agent's keepalive loop calls this every
        ~4 seconds when the session has been silent for 3.5+ seconds.

        Why not send_audio(b"")?
            Deepgram treats an empty binary frame differently from a JSON
            KeepAlive — empty binary may be interpreted as end-of-stream on
            some model configurations. The explicit JSON KeepAlive is the
            documented approach for Nova-2.

        Raises:
            DeepgramConnectionError: If not connected or the send fails.
                The STT agent's keepalive loop exits on this exception.
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
            raise DeepgramConnectionError(
                f"Deepgram keepalive send failed: {exc}"
            ) from exc

    # ── Receive loop ──────────────────────────────────────────────────────────

    async def _receive_loop(self) -> None:
        """Receive and dispatch Deepgram messages for the lifetime of the connection.

        Handles four message types:
            Results      — transcript (partial or final) → fires on_transcript
            UtteranceEnd — end of speech segment → fires on_utterance_end
            Metadata     — logged at debug, otherwise ignored
            Error        — logged at error level
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
                        if self._on_utterance_end is not None:
                            await self._on_utterance_end()

                    elif msg_type == "Metadata":
                        logger.debug("Deepgram metadata: %s", data)

                    elif msg_type == "Error":
                        logger.error(
                            "Deepgram error message: code=%s description=%s",
                            data.get("error_code", "unknown"),
                            data.get("description", "no description"),
                        )

                    elif msg_type == "SpeechStarted":
                        logger.debug("Deepgram SpeechStarted")

                    else:
                        logger.debug(
                            "Deepgram unrecognised message type: %r", msg_type
                        )

                except json.JSONDecodeError:
                    logger.warning(
                        "Deepgram sent non-JSON message — ignoring (len=%d)",
                        len(raw_message)
                        if isinstance(raw_message, (str, bytes))
                        else -1,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception("Unhandled error processing Deepgram message")

        except asyncio.CancelledError:
            logger.debug("Deepgram receive loop cancelled")
            raise  # Must re-raise so the task is marked cancelled correctly.

        except ConnectionClosed as exc:
            logger.warning("Deepgram connection closed in receive loop: %s", exc)
            if not self._closing:
                self._connected = False
                await self._try_reconnect()

        except Exception:
            logger.exception("Deepgram receive loop encountered unexpected error")
            self._connected = False

    async def _handle_results(self, data: dict[str, Any]) -> None:
        """Parse a Deepgram Results message and fire the transcript callback.

        Silently drops empty transcripts — common at utterance boundaries and
        at connection start before the model has warmed up.
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

            # confidence may be missing on some Deepgram model configurations.
            try:
                confidence = float(best.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0

            is_final = bool(data.get("is_final", False))
            speech_final = bool(data.get("speech_final", False))

            result = TranscriptResult(
                type="transcript_final" if (is_final or speech_final) else "transcript_partial",
                text=text,
                confidence=confidence,
                is_final=is_final or speech_final,
            )
            await self._on_transcript(result)

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Error parsing Deepgram Results message")

    # ── Reconnection ──────────────────────────────────────────────────────────

    async def _try_reconnect(self) -> None:
        """Attempt to reconnect after an unexpected connection drop.

        Uses linear back-off: RECONNECT_DELAY_SECONDS × attempt number
        (2s, 4s, 6s). Gives up after MAX_RECONNECT_ATTEMPTS.

        Does nothing if close() has been called (_closing is True) because
        the drop is intentional in that case.

        After a successful reconnect, connect() resets _reconnect_attempts
        to 0 so future drops get the full retry budget again.
        """
        if self._closing:
            return

        if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            logger.error(
                "Deepgram: giving up after %d failed reconnect attempt(s)",
                MAX_RECONNECT_ATTEMPTS,
            )
            return

        self._reconnect_attempts += 1
        delay = RECONNECT_DELAY_SECONDS * self._reconnect_attempts

        logger.info(
            "Deepgram: reconnect attempt %d/%d in %.1fs",
            self._reconnect_attempts,
            MAX_RECONNECT_ATTEMPTS,
            delay,
        )

        await asyncio.sleep(delay)

        try:
            await self.connect()
            logger.info(
                "Deepgram: reconnected successfully on attempt %d",
                self._reconnect_attempts,
            )
        except DeepgramConnectionError as exc:
            logger.warning(
                "Deepgram: reconnect attempt %d failed: %s",
                self._reconnect_attempts,
                exc,
            )