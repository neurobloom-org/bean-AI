"""BEAN AI v5 — ElevenLabs TTS service.

Privacy: Audio is generated on-demand and streamed chunk-by-chunk directly
to the ESP32 WebSocket. It is NEVER written to Supabase Storage or any disk.
"""

import logging
from fastapi import WebSocket
from elevenlabs import AsyncElevenLabs
from shared.config import get_settings
from shared.exceptions import ElevenLabsError

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
    """Generate TTS for text and stream audio chunks to the ESP32 WebSocket.

    Chunks are sent as binary WebSocket frames with a JSON header prefix.
    Audio is never stored — it exists only in transit.
    """
    if not text.strip():
        return

    settings = get_settings()
    client = _get_client()

    try:
        audio_stream = await client.generate(
            text=text,
            voice=settings.elevenlabs_voice_id,
            model=settings.elevenlabs_model_id,
            stream=True,
        )

        chunk_count = 0
        async for chunk in audio_stream:
            if chunk:
                # Send audio chunk as binary frame
                # ESP32 identifies it as audio by the "response_audio" prefix message
                await websocket.send_bytes(chunk)
                chunk_count += 1

        # Signal end of audio for this turn
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
