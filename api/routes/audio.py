"""BEAN AI — Streaming TTS audio endpoint for ESP32.

ESP32-audioI2S uses audio.connecttohost(url) which does chunked HTTP
streaming. This endpoint pipes ElevenLabs MP3 chunks to the ESP32 as
they arrive — the ESP32 starts playing within ~200ms of the first chunk.

The server sends the URL over WebSocket before TTS starts, so the ESP32
can connect immediately while the server is still generating audio.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from services.audio_stream_store import drain_queue, get_stream

router = APIRouter()


@router.get("/audio/stream/{token}")
async def stream_audio(token: str) -> StreamingResponse:
    """Chunked MP3 stream for ESP32 playback via audio.connecttohost()."""
    if get_stream(token) is None:
        raise HTTPException(status_code=404, detail="Stream not found or expired")

    return StreamingResponse(
        drain_queue(token),
        media_type="audio/mpeg",
        headers={"Cache-Control": "no-cache"},
    )
