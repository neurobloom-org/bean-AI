"""
services/deepgram_service.py
=============================
Deepgram speech-to-text service.
Only needs shared/ — no other services needed.
"""

from deepgram import DeepgramClient, PrerecordedOptions

from shared.config import config

deepgram = DeepgramClient(config.DEEPGRAM_API_KEY)


async def transcribe_audio(audio_bytes: bytes) -> str:
    """Convert audio bytes to text using Deepgram."""
    payload = {"buffer": audio_bytes}
    options = PrerecordedOptions(
        model="nova-2",
        language="en",
        smart_format=True,
    )
    response = await deepgram.listen.asyncprerecorded.v("1").transcribe_file(
        payload, options
    )
    return response.results.channels[0].alternatives[0].transcript


async def transcribe_url(audio_url: str) -> str:
    """Convert audio from URL to text using Deepgram."""
    payload = {"url": audio_url}
    options = PrerecordedOptions(
        model="nova-2",
        language="en",
        smart_format=True,
    )
    response = await deepgram.listen.asyncprerecorded.v("1").transcribe_url(
        payload, options
    )
    return response.results.channels[0].alternatives[0].transcript
