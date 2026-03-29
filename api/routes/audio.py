"""BEAN AI — TTS audio download endpoint for ESP32.

The ESP32's ESP32-audioI2S library cannot decode streaming binary WebSocket
frames. Instead, the server synthesises TTS audio, stores it briefly in RAM,
and sends the robot a JSON message with a one-time download URL:

    {"type": "play_audio", "url": "https://.../audio/<token>", "format": "mp3"}

The robot fetches the URL over plain HTTP, plays it, and the entry is deleted.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from services.audio_store import pop_audio

router = APIRouter()


@router.get("/audio/{token}")
async def serve_audio(token: str) -> Response:
    """One-time MP3 audio download for ESP32 playback."""
    audio = pop_audio(token)
    if audio is None:
        raise HTTPException(status_code=404, detail="Audio not found or expired")
    return Response(
        content=audio,
        media_type="audio/mpeg",
        headers={"Content-Disposition": "inline"},
    )
