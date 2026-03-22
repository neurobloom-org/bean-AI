"""BEAN AI v1 — ElevenLabs TTS service.

Privacy: Audio is generated on-demand and streamed chunk-by-chunk directly
to the ESP32 WebSocket. It is NEVER written to Supabase Storage or any disk.

Exported functions:
    stream_tts_to_websocket(text, websocket, turn_id, voice_id)
        Stream audio binary frames directly to a live WebSocket connection.
        Used by the WebSocket handler for direct robot playback.

    synthesize_speech_full(text, turn_id, voice_id) -> bytes
        Accumulate all TTS audio and return it as a single bytes object.
        Used by the TTS agent's full-audio (cache-backed) path.

    synthesize_speech_stream(text, turn_id, voice_id) -> AsyncGenerator[TTSChunk]
        Yield TTSChunk objects as audio arrives from ElevenLabs.
        True streaming — first chunk arrives as soon as ElevenLabs starts
        generating, not after the full response is buffered.
        Used by the TTS agent's stream_tts_chunks() path.

    _reset_client()
        Test helper — clears the cached AsyncElevenLabs client so tests can
        patch settings without restarting the process.

voice_id resolution (same rule for all three functions):
    If voice_id is provided and non-empty → use it.
    Otherwise → fall back to settings.elevenlabs_voice_id.
    If neither is set → raise ElevenLabsError.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Final

from elevenlabs import AsyncElevenLabs
from fastapi import WebSocket

from shared.config import get_settings
from shared.exceptions import ElevenLabsError
from shared.schemas import TTSChunk

__all__ = [
    "stream_tts_to_websocket",
    "synthesize_speech_full",
    "synthesize_speech_stream",
    "_reset_client",
]

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Timeout for the first chunk to arrive from ElevenLabs.
# If ElevenLabs doesn't respond within this window the call is considered hung.
_FIRST_CHUNK_TIMEOUT_SECONDS: Final[float] = 10.0

# Per-chunk timeout for subsequent chunks during streaming.
# ElevenLabs sends chunks in rapid succession once started; a long gap means
# the stream has stalled.
_CHUNK_TIMEOUT_SECONDS: Final[float] = 8.0

# Timeout for accumulating the full audio (synthesize_speech_full).
_FULL_AUDIO_TIMEOUT_SECONDS: Final[float] = 30.0

# Timeout for direct WebSocket streaming (stream_tts_to_websocket).
_WS_STREAM_TIMEOUT_SECONDS: Final[float] = 30.0

# ── Singleton client ──────────────────────────────────────────────────────────

_client: AsyncElevenLabs | None = None


def _get_client() -> AsyncElevenLabs:
    """Return a cached AsyncElevenLabs client, creating one on first call."""
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.elevenlabs_api_key:
            raise ElevenLabsError("ELEVENLABS_API_KEY is not configured")
        _client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)
    return _client


def _reset_client() -> None:
    """Clear the cached client. Use in tests after patching settings."""
    global _client
    _client = None


# ── Voice ID resolution ───────────────────────────────────────────────────────


def _resolve_voice_id(voice_id: str | None) -> str:
    """Return the effective ElevenLabs voice ID.

    Priority:
        1. voice_id argument (if non-empty after stripping)
        2. settings.elevenlabs_voice_id

    Raises:
        ElevenLabsError: If neither source provides a non-empty voice ID.
    """
    if isinstance(voice_id, str) and voice_id.strip():
        return voice_id.strip()

    settings = get_settings()
    configured = getattr(settings, "elevenlabs_voice_id", None)

    if isinstance(configured, str) and configured.strip():
        return configured.strip()

    raise ElevenLabsError(
        "No ElevenLabs voice ID available. Provide voice_id explicitly "
        "or set ELEVENLABS_VOICE_ID in your environment."
    )


# ── Direct WebSocket streaming ────────────────────────────────────────────────


async def stream_tts_to_websocket(
    text: str,
    websocket: WebSocket,
    turn_id: str,
    voice_id: str | None = None,
) -> None:
    """Generate TTS and stream audio chunks directly to the ESP32 WebSocket.

    Chunks are sent as binary WebSocket frames as they arrive from ElevenLabs
    (true streaming — not buffered). A JSON sentinel is sent after the last
    audio frame so the ESP32 knows playback is complete.

    Audio is never stored — it exists only in transit.

    Args:
        text:     Text to synthesize. Must not be empty.
        websocket: Active WebSocket connection to the ESP32.
        turn_id:  Turn identifier for log correlation.
        voice_id: ElevenLabs voice ID override. Falls back to settings.

    Raises:
        ElevenLabsError: If synthesis fails or the stream stalls.
        ValueError:      If text is empty.
    """
    if not text or not text.strip():
        raise ValueError("stream_tts_to_websocket: text must not be empty")

    effective_voice_id = _resolve_voice_id(voice_id)
    settings = get_settings()
    client = _get_client()

    try:
        chunk_count = 0

        async def _stream() -> None:
            nonlocal chunk_count
            async for chunk in client.text_to_speech.convert(
                voice_id=effective_voice_id,
                text=text,
                model_id=settings.elevenlabs_model_id,
            ):
                if chunk:
                    await websocket.send_bytes(chunk)
                    chunk_count += 1

        await asyncio.wait_for(_stream(), timeout=_WS_STREAM_TIMEOUT_SECONDS)

        await websocket.send_json(
            {
                "type": "response_audio_end",
                "turn_id": turn_id,
            }
        )
        logger.debug(
            "TTS streamed to WS: %d chunks turn=%s voice=%s",
            chunk_count,
            turn_id[:8],
            effective_voice_id[:8],
        )

    except asyncio.TimeoutError as exc:
        logger.error(
            "TTS WebSocket stream timed out after %.1fs turn=%s",
            _WS_STREAM_TIMEOUT_SECONDS,
            turn_id[:8],
        )
        raise ElevenLabsError(
            f"TTS WebSocket stream timed out after {_WS_STREAM_TIMEOUT_SECONDS}s"
        ) from exc

    except Exception as exc:
        logger.error("TTS WebSocket stream failed turn=%s: %s", turn_id[:8], exc)
        raise ElevenLabsError(f"TTS WebSocket stream failed: {exc}") from exc


# ── Full-audio synthesis ──────────────────────────────────────────────────────


async def synthesize_speech_full(
    text: str,
    turn_id: str,
    voice_id: str | None = None,
) -> bytes:
    """Generate complete TTS audio and return it as raw bytes.

    Accumulates all chunks from ElevenLabs into a single bytes object.
    Use for cache-backed flows where the full audio is needed upfront.
    For low-latency robot playback, use synthesize_speech_stream() instead.

    Audio is never stored — exists only in memory.

    Args:
        text:     Text to synthesize. Must not be empty.
        turn_id:  Turn identifier for log correlation.
        voice_id: ElevenLabs voice ID override. Falls back to settings.

    Returns:
        Raw audio bytes (MP3 or PCM depending on ElevenLabs model).

    Raises:
        ElevenLabsError: If synthesis fails, times out, or returns empty audio.
        ValueError:      If text is empty.
    """
    if not text or not text.strip():
        raise ValueError("synthesize_speech_full: text must not be empty")

    effective_voice_id = _resolve_voice_id(voice_id)
    settings = get_settings()
    client = _get_client()

    try:
        audio_bytes = b""

        async def _accumulate() -> bytes:
            nonlocal audio_bytes
            async for chunk in client.text_to_speech.convert(
                voice_id=effective_voice_id,
                text=text,
                model_id=settings.elevenlabs_model_id,
            ):
                if chunk:
                    audio_bytes += chunk
            return audio_bytes

        result = await asyncio.wait_for(
            _accumulate(), timeout=_FULL_AUDIO_TIMEOUT_SECONDS
        )

        if not result:
            raise ElevenLabsError("ElevenLabs returned empty audio — no bytes received")

        logger.debug(
            "TTS full: %d bytes turn=%s voice=%s",
            len(result),
            turn_id[:8],
            effective_voice_id[:8],
        )
        return result

    except asyncio.TimeoutError as exc:
        logger.error(
            "TTS full synthesis timed out after %.1fs turn=%s",
            _FULL_AUDIO_TIMEOUT_SECONDS,
            turn_id[:8],
        )
        raise ElevenLabsError(
            f"TTS full synthesis timed out after {_FULL_AUDIO_TIMEOUT_SECONDS}s"
        ) from exc

    except ElevenLabsError:
        raise

    except Exception as exc:
        logger.error("TTS full synthesis failed turn=%s: %s", turn_id[:8], exc)
        raise ElevenLabsError(f"TTS full synthesis failed: {exc}") from exc


# ── Streaming synthesis ───────────────────────────────────────────────────────


async def synthesize_speech_stream(
    text: str,
    turn_id: str,
    voice_id: str | None = None,
) -> AsyncGenerator[TTSChunk, None]:
    """Yield TTSChunk objects as audio arrives from ElevenLabs.

    True streaming: the first chunk is yielded as soon as ElevenLabs starts
    generating audio, not after the full response is buffered. This is critical
    for low-latency robot playback — the ESP32 can start playing immediately.

    Uses a look-ahead of one chunk to correctly set is_final=True on the last
    chunk without buffering the entire response.

    Audio is never stored — exists only in transit.

    Args:
        text:     Text to synthesize. Must not be empty.
        turn_id:  Turn identifier for log correlation and TTSChunk fields.
        voice_id: ElevenLabs voice ID override. Falls back to settings.

    Yields:
        TTSChunk instances. The last chunk has is_final=True.
        All others have is_final=False.

    Raises:
        ElevenLabsError: If synthesis fails, the first chunk times out,
                         or a subsequent chunk stalls beyond _CHUNK_TIMEOUT_SECONDS.
        ValueError:      If text is empty or no audio chunks are received.
    """
    if not text or not text.strip():
        raise ValueError("synthesize_speech_stream: text must not be empty")

    effective_voice_id = _resolve_voice_id(voice_id)
    settings = get_settings()
    client = _get_client()

    try:
        stream = client.text_to_speech.convert(
            voice_id=effective_voice_id,
            text=text,
            model_id=settings.elevenlabs_model_id,
        )
        stream_iter = stream.__aiter__()

        # ── Await the first chunk with a tighter timeout ──────────────────────
        # ElevenLabs has a startup latency before any audio arrives.
        # A tight first-chunk timeout catches hung connections early.
        try:
            first_raw: bytes | None = None
            while first_raw is None:
                raw = await asyncio.wait_for(
                    stream_iter.__anext__(),
                    timeout=_FIRST_CHUNK_TIMEOUT_SECONDS,
                )
                if raw:
                    first_raw = raw
        except StopAsyncIteration:
            raise ValueError(
                "synthesize_speech_stream: ElevenLabs returned no audio chunks"
            ) from None
        except asyncio.TimeoutError as exc:
            raise ElevenLabsError(
                f"TTS stream: first chunk did not arrive within "
                f"{_FIRST_CHUNK_TIMEOUT_SECONDS}s"
            ) from exc

        # ── Stream remaining chunks with look-ahead for is_final ──────────────
        # prev_chunk holds the chunk we received last iteration.
        # We yield it when we receive the NEXT chunk (so we know it's not final).
        # When the stream ends, we yield prev_chunk with is_final=True.
        prev_chunk: bytes = first_raw

        try:
            while True:
                try:
                    raw = await asyncio.wait_for(
                        stream_iter.__anext__(),
                        timeout=_CHUNK_TIMEOUT_SECONDS,
                    )
                except StopAsyncIteration:
                    # Stream exhausted — prev_chunk is the last one.
                    break
                except asyncio.TimeoutError as exc:
                    raise ElevenLabsError(
                        f"TTS stream stalled: no chunk received within "
                        f"{_CHUNK_TIMEOUT_SECONDS}s"
                    ) from exc

                if raw:
                    # prev_chunk is not final — yield it and advance.
                    yield TTSChunk(
                        audio_chunk=prev_chunk,
                        is_final=False,
                        turn_id=turn_id,
                    )
                    prev_chunk = raw
                # empty raw chunks from ElevenLabs are ignored (keepalives)

        finally:
            # this correctly marks the single chunk as final.
            pass

        yield TTSChunk(
            audio_chunk=prev_chunk,
            is_final=True,
            turn_id=turn_id,
        )

        logger.debug(
            "TTS stream complete turn=%s voice=%s",
            turn_id[:8],
            effective_voice_id[:8],
        )

    except (ElevenLabsError, ValueError):
        raise

    except Exception as exc:
        logger.error("TTS stream failed turn=%s: %s", turn_id[:8], exc)
        raise ElevenLabsError(f"TTS stream failed: {exc}") from exc
