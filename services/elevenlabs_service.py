"""BEAN AI v5 — ElevenLabs TTS service.

Privacy: Audio is generated on-demand and streamed chunk-by-chunk directly
to the ESP32 WebSocket. It is NEVER written to Supabase Storage or any disk.
"""

import logging

from elevenlabs import AsyncElevenLabs
from fastapi import WebSocket

from shared.config import get_settings
from shared.exceptions import ElevenLabsError
from shared.schemas import TTSChunk

logger = logging.getLogger(__name__)

_client: AsyncElevenLabs | None = None


def _get_client() -> AsyncElevenLabs:
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncElevenLabs(api_key=settings.elevenlabs_api_key)
    return _client


async def stream_tts_to_websocket(
    text: str,
    websocket: WebSocket,
    turn_id: str,
) -> None:
    """Generate TTS and stream audio chunks directly to the ESP32 WebSocket.

    Chunks are sent as binary WebSocket frames.
    Audio is never stored — it exists only in transit.
    """
    if not text.strip():
        return

    settings = get_settings()
    client = _get_client()

    try:
        chunk_count = 0

        # convert() on AsyncElevenLabs is an async generator — iterate with async for
        async for chunk in client.text_to_speech.convert(
            voice_id=settings.elevenlabs_voice_id,
            text=text,
            model_id=settings.elevenlabs_model_id,
        ):
            if chunk:
                await websocket.send_bytes(chunk)
                chunk_count += 1

        await websocket.send_json(
            {
                "type": "response_audio_end",
                "turn_id": turn_id,
            }
        )
        logger.debug("TTS streamed: %d chunks for turn %s", chunk_count, turn_id[:8])

    except Exception as exc:
        logger.error("TTS streaming failed: %s", exc)
        raise ElevenLabsError(f"TTS failed: {exc}") from exc


async def synthesize_speech_full(text: str, turn_id: str) -> bytes:
    """Generate complete TTS audio and return as raw bytes.

    Audio is never stored — exists only in memory.
    """
    if not text.strip():
        return b""

    settings = get_settings()
    client = _get_client()

    try:
        audio_bytes = b""

        # convert() is an async generator — accumulate all chunks into bytes
        async for chunk in client.text_to_speech.convert(
            voice_id=settings.elevenlabs_voice_id,
            text=text,
            model_id=settings.elevenlabs_model_id,
        ):
            if chunk:
                audio_bytes += chunk

        logger.debug("TTS full: %d bytes for turn %s", len(audio_bytes), turn_id[:8])
        return audio_bytes

    except Exception as exc:
        logger.error("TTS full generation failed: %s", exc)
        raise ElevenLabsError(f"TTS failed: {exc}") from exc


async def synthesize_speech_stream(
    text: str,
    turn_id: str,
):
    """Async generator that yields TTSChunk objects.

    Used for streaming audio directly to the ESP32.
    Audio is never stored — exists only in transit.

    Yields:
        TTSChunk with audio_chunk bytes and is_final flag.
    """
    if not text.strip():
        return

    settings = get_settings()
    client = _get_client()

    try:
        # Collect all chunks first so we can mark the last one as is_final
        chunks = []
        async for chunk in client.text_to_speech.convert(
            voice_id=settings.elevenlabs_voice_id,
            text=text,
            model_id=settings.elevenlabs_model_id,
        ):
            if chunk:
                chunks.append(chunk)

        total = len(chunks)
        for i, chunk in enumerate(chunks):
            yield TTSChunk(
                audio_chunk=chunk,
                is_final=(i == total - 1),
                turn_id=turn_id,
            )

    except Exception as exc:
        logger.error("TTS stream failed: %s", exc)
        raise ElevenLabsError(f"TTS stream failed: {exc}") from exc
