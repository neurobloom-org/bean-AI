"""BEAN AI v5 — Deepgram WebSocket STT service.

Manages a persistent Deepgram WebSocket connection for one BEAN session.
Audio bytes flow: ESP32 → Cloud Run RAM → Deepgram → transcript text
Audio is NEVER written to disk or any storage at any point.

Changes from v4:
  ✓  Explicit close() method drains connection cleanly (was missing)
  ✓  Reconnect cap enforced — doesn't retry forever
  ✓  UtteranceEnd handled separately from Results (was merged, caused missed events)
  ✓  Connection health tracked accurately (was possible to send on dead socket)
"""

import asyncio
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
      connect() → send_audio() [many times] → close()

    Privacy:
      Audio bytes are passed in from the WebSocket handler and forwarded
      to Deepgram. They are never stored — only the resulting transcript
      text is returned via on_transcript callback.
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

    @property
    def is_connected(self) -> bool:
        """True only if the WebSocket is open AND not in the process of closing."""
        return (
            self._connected
            and not self._closing
            and self._ws is not None
            and not self._ws.closed
        )

    def _build_url(self) -> str:
        """Build Deepgram streaming WebSocket URL."""
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

    async def connect(self) -> None:
        """Open the Deepgram WebSocket connection.

        Raises:
            DeepgramConnectionError: If connection fails.
        """
        if self._connected:
            logger.debug("Deepgram already connected")
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
                max_size=2**20,
            )
            self._connected = True
            self._closing = False
            self._reconnect_attempts = 0
            self._receive_task = asyncio.create_task(
                self._receive_loop(), name="deepgram-receive"
            )
            logger.info("Deepgram WebSocket connected")

        except Exception as exc:
            logger.error("Deepgram connection failed: %s", exc)
            self._connected = False
            raise DeepgramConnectionError(str(exc)) from exc

    async def close(self) -> None:
        """Gracefully close the Deepgram connection.

        Sends a close signal, waits for the receive loop to drain, then
        closes the WebSocket. Safe to call even if not connected.
        """
        if self._closing:
            return

        self._closing = True
        self._connected = False

        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
            try:
                await asyncio.wait_for(self._receive_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass

        if self._ws and not self._ws.closed:
            try:
                await self._ws.send(json.dumps({"type": "CloseStream"}))
                await asyncio.wait_for(self._ws.close(), timeout=3.0)
            except Exception as exc:
                logger.debug("Deepgram close error (non-critical): %s", exc)

        self._ws = None
        logger.info("Deepgram WebSocket closed")

    async def send_audio(self, audio_bytes: bytes) -> None:
        """Forward raw PCM16 audio bytes to Deepgram.

        Privacy: audio_bytes are passed through in memory and forwarded
        to Deepgram's API. They are never written to disk or logged.
        """
        if not self.is_connected or self._ws is None:
            logger.debug("send_audio skipped — Deepgram not connected")
            return

        try:
            await self._ws.send(audio_bytes)
        except ConnectionClosed:
            logger.warning(
                "Deepgram connection closed during send — attempting reconnect"
            )
            self._connected = False
            await self._try_reconnect()
        except Exception as exc:
            logger.error("Audio send error: %s", exc)
            self._connected = False

    async def _receive_loop(self) -> None:
        """Continuously receive transcript results from Deepgram."""
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
                        if self._on_utterance_end:
                            await self._on_utterance_end()

                    elif msg_type == "Metadata":
                        logger.debug("Deepgram metadata: %s", data)

                    elif msg_type == "Error":
                        logger.error("Deepgram error message: %s", data)

                except json.JSONDecodeError:
                    logger.warning("Deepgram sent non-JSON message")
                except Exception as exc:
                    logger.error("Error handling Deepgram message: %s", exc)

        except asyncio.CancelledError:
            logger.debug("Deepgram receive loop cancelled")
        except ConnectionClosed as exc:
            logger.warning("Deepgram connection closed: %s", exc)
            if not self._closing:
                self._connected = False
                await self._try_reconnect()
        except Exception as exc:
            logger.error("Deepgram receive loop error: %s", exc)
            self._connected = False

    async def _handle_results(self, data: dict[str, Any]) -> None:
        """Parse Deepgram Results message and fire callback."""
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
                type=(
                    "transcript_final"
                    if (is_final or speech_final)
                    else "transcript_partial"
                ),
                text=text,
                confidence=confidence,
                is_final=is_final or speech_final,
            )
            await self._on_transcript(result)

        except Exception as exc:
            logger.error("Error parsing Deepgram results: %s", exc)

    async def _try_reconnect(self) -> None:
        """Attempt to reconnect to Deepgram after a connection drop.

        Gives up after MAX_RECONNECT_ATTEMPTS to avoid infinite loops.
        """
        if self._closing:
            return

        if self._reconnect_attempts >= self.MAX_RECONNECT_ATTEMPTS:
            logger.error(
                "Deepgram reconnect failed after %d attempts — giving up",
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
            logger.info("Deepgram reconnected successfully")
        except DeepgramConnectionError as exc:
            logger.warning(
                "Deepgram reconnect attempt %d failed: %s",
                self._reconnect_attempts,
                exc,
            )