"""BEAN AI — Per-turn streaming audio queue.

Bridges the TTS generator (ElevenLabs) and the ESP32's HTTP streaming
connection via an asyncio.Queue.

Flow:
    1. _process_turn creates a queue and sends the URL to the ESP32.
    2. ElevenLabs chunks are put() into the queue as they arrive.
    3. GET /audio/stream/{token} reads from the queue and yields chunks
       to the ESP32 as a chunked HTTP response.
    4. ESP32-audioI2S calls audio.connecttohost(url) and plays immediately.

The queue decouples producer (TTS) from consumer (ESP32 HTTP connection):
    - If TTS is ahead: chunks wait in the queue (bounded at 50).
    - If ESP32 is ahead: the HTTP handler waits on queue.get().
A None sentinel signals end-of-stream.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator

__all__ = ["create_audio_stream", "get_stream", "remove_stream"]

# token → Queue[bytes | None]  (None = end-of-stream sentinel)
_streams: dict[str, asyncio.Queue[bytes | None]] = {}


def create_audio_stream() -> tuple[str, asyncio.Queue[bytes | None]]:
    """Create a new streaming queue. Returns (token, queue)."""
    token = str(uuid.uuid4())
    _streams[token] = asyncio.Queue(maxsize=50)
    return token, _streams[token]


def get_stream(token: str) -> asyncio.Queue[bytes | None] | None:
    return _streams.get(token)


def remove_stream(token: str) -> None:
    _streams.pop(token, None)


async def drain_queue(token: str) -> AsyncGenerator[bytes, None]:
    """Async generator that yields chunks from the queue until sentinel."""
    q = get_stream(token)
    if q is None:
        return
    try:
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=30.0)
            except asyncio.TimeoutError:
                break
            if chunk is None:
                break
            yield chunk
    finally:
        remove_stream(token)
