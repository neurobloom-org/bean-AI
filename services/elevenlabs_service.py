"""
services/elevenlabs_service.py
===============================
ElevenLabs text-to-speech service.
Only needs shared/ — no other services needed.
"""

from elevenlabs import AsyncElevenLabs

from shared.config import config

elevenlabs = AsyncElevenLabs(api_key=config.ELEVENLABS_API_KEY)


async def text_to_speech(text: str) -> bytes:
    """Convert text to speech audio bytes using ElevenLabs."""
    audio = await elevenlabs.generate(
        text=text,
        voice=config.ELEVENLABS_VOICE_ID,
        model="eleven_turbo_v2",
    )
    audio_bytes = b""
    async for chunk in audio:
        audio_bytes += chunk
    return audio_bytes


async def get_voices() -> list[dict]:
    """Get list of available voices."""
    voices = await elevenlabs.voices.get_all()
    return [{"id": v.voice_id, "name": v.name} for v in voices.voices]
