"""BEAN AI — In-memory TTS audio store.

Holds freshly-synthesised TTS MP3 bytes keyed by a one-time UUID token.
The ESP32 fetches the audio via GET /audio/{token} and the entry is deleted
on first read (or after TTL expiry).

Privacy: audio is never written to disk or Supabase Storage.
"""

from __future__ import annotations

import time
import uuid

__all__ = ["store_audio", "pop_audio", "cleanup_expired"]

# token → (mp3_bytes, expires_at_monotonic)
_store: dict[str, tuple[bytes, float]] = {}

_TTL_SECONDS: float = 120.0  # 2 minutes — plenty for the ESP32 to fetch


def store_audio(audio_bytes: bytes) -> str:
    """Store MP3 bytes under a fresh UUID token. Returns the token."""
    token = str(uuid.uuid4())
    _store[token] = (audio_bytes, time.monotonic() + _TTL_SECONDS)
    return token


def pop_audio(token: str) -> bytes | None:
    """Return and delete the audio for token, or None if missing/expired."""
    entry = _store.pop(token, None)
    if entry is None:
        return None
    audio_bytes, expires_at = entry
    if time.monotonic() > expires_at:
        return None
    return audio_bytes


def cleanup_expired() -> None:
    """Remove all expired entries (called periodically by the purge loop)."""
    now = time.monotonic()
    expired = [k for k, (_, exp) in list(_store.items()) if now > exp]
    for k in expired:
        _store.pop(k, None)
