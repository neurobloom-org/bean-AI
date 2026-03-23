"""BEAN AI — Music streaming service.

Fetches song metadata from Supabase and streams audio from Supabase Storage
bucket "bean-music" directly to the ESP32 WebSocket as binary frames.

Flow:
  1. pick_song(user_id, mood)
         Query user_songs table → random song matching mood.
         Prefers user-uploaded songs, falls back to default songs.

  2. stream_song_chunks(storage_path)
         Create a short-lived signed URL for the storage path.
         Stream the audio file in CHUNK_SIZE chunks via httpx.
         Yields bytes chunks as they arrive (true streaming, not buffered).

  3. get_song_count(user_id, mood)
         Returns total songs available for a mood (user + defaults).

Privacy:
  - Audio is never written to disk or any database.
  - Signed URLs expire in SIGNED_URL_TTL_SECONDS (default 5 min).
  - Only the service-role client is used here (server-side streaming).
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import random
import uuid
from collections.abc import AsyncGenerator
from typing import Any, Final

import httpx

from services.supabase_client import get_service_client
from shared.config import get_settings

__all__ = [
    "pick_song",
    "stream_song_chunks",
    "get_songs_for_mood",
    "get_song_count",
    "delete_user_song",
    "register_uploaded_song",
    "STORAGE_BUCKET",
    "VALID_MOODS",
]

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

STORAGE_BUCKET: Final[str] = "bean-music"

#: Audio chunk size streamed to ESP32 per binary WebSocket frame.
#: 8 KB is a good balance — small enough to interleave with other messages,
#: large enough to keep the stream efficient.
CHUNK_SIZE: Final[int] = 8 * 1024  # 8 KB

#: Signed URL lifetime. Keep short so leaked URLs expire quickly.
SIGNED_URL_TTL_SECONDS: Final[int] = 300  # 5 minutes

#: Timeout for the entire signed-URL fetch from Supabase.
_SIGNED_URL_TIMEOUT: Final[float] = 5.0

#: Total stream timeout — largest expected file (20 MB) at worst-case bandwidth.
_STREAM_TIMEOUT: Final[float] = 120.0

#: DB query timeout.
_DB_TIMEOUT: Final[float] = 5.0

#: Max songs returned in a list query.
_MAX_LIST: Final[int] = 100

VALID_MOODS: Final[frozenset[str]] = frozenset(
    {"calm", "happy", "sad", "lofi", "nature", "classical"}
)

#: Max file size allowed for user uploads (20 MB).
MAX_UPLOAD_BYTES: Final[int] = 20 * 1024 * 1024

ALLOWED_MIME_TYPES: Final[frozenset[str]] = frozenset(
    {"audio/mpeg", "audio/ogg", "audio/wav", "audio/mp4", "audio/aac"}
)


# ── Song selection ────────────────────────────────────────────────────────────


async def pick_song(user_id: str, mood: str) -> dict[str, Any] | None:
    """Pick a random song for the given mood.

    Selection order:
        1. User's own songs for this mood (shuffled)
        2. Default songs for this mood (shuffled)
        3. Any calm default song (ultimate fallback)

    Returns the song metadata dict, or None if no songs exist at all.
    """
    mood = mood.lower().strip()
    if mood not in VALID_MOODS:
        mood = "calm"

    songs = await get_songs_for_mood(user_id, mood)
    if songs:
        return random.choice(songs)

    # Cross-mood fallback: try calm defaults
    if mood != "calm":
        logger.debug("No songs found for mood=%s, falling back to calm", mood)
        fallback = await _get_default_songs("calm")
        if fallback:
            return random.choice(fallback)

    logger.warning(
        "No songs found for user=%s mood=%s and no defaults available",
        user_id[:8],
        mood,
    )
    return None


async def get_songs_for_mood(user_id: str, mood: str) -> list[dict[str, Any]]:
    """Return all songs available to this user for the given mood.

    Combines user-uploaded songs and default (shared) songs.
    """
    mood = mood.lower().strip()
    if mood not in VALID_MOODS:
        return []

    try:
        client = await get_service_client()

        result = await asyncio.wait_for(
            client.table("user_songs")
            .select("id, title, mood, storage_path, mime_type, duration_seconds, is_default")
            .eq("mood", mood)
            .or_(f"user_id.eq.{user_id},is_default.eq.true")
            .limit(_MAX_LIST)
            .execute(),
            timeout=_DB_TIMEOUT,
        )

        return result.data or []

    except asyncio.TimeoutError:
        logger.error("get_songs_for_mood timed out [user=%s mood=%s]", user_id[:8], mood)
        return []
    except Exception as exc:
        logger.error("get_songs_for_mood failed: %s", exc)
        return []


async def get_song_count(user_id: str, mood: str) -> int:
    """Return total number of songs available for this user/mood."""
    songs = await get_songs_for_mood(user_id, mood)
    return len(songs)


async def _get_default_songs(mood: str) -> list[dict[str, Any]]:
    """Fetch only default (admin-seeded) songs for a mood."""
    try:
        client = await get_service_client()
        result = await asyncio.wait_for(
            client.table("user_songs")
            .select("id, title, mood, storage_path, mime_type, duration_seconds")
            .eq("mood", mood)
            .eq("is_default", True)
            .execute(),
            timeout=_DB_TIMEOUT,
        )
        return result.data or []
    except Exception as exc:
        logger.error("_get_default_songs failed: %s", exc)
        return []


# ── Signed URL ────────────────────────────────────────────────────────────────


async def _get_signed_url(storage_path: str) -> str:
    """Create a short-lived signed download URL for a storage path.

    Raises:
        RuntimeError: If Supabase returns an error or no URL.
    """
    try:
        client = await get_service_client()

        result = await asyncio.wait_for(
            client.storage.from_(STORAGE_BUCKET).create_signed_url(
                storage_path,
                expires_in=SIGNED_URL_TTL_SECONDS,
            ),
            timeout=_SIGNED_URL_TIMEOUT,
        )

        # Supabase SDK returns {"signedURL": "...", "error": null}
        signed_url: str | None = None
        if isinstance(result, dict):
            signed_url = result.get("signedURL") or result.get("signed_url")
        elif hasattr(result, "signed_url"):
            signed_url = result.signed_url

        if not signed_url:
            raise RuntimeError(
                f"Supabase returned no signed URL for path={storage_path!r}"
            )

        return signed_url

    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"Signed URL fetch timed out after {_SIGNED_URL_TIMEOUT}s "
            f"for path={storage_path!r}"
        ) from exc


# ── Streaming ─────────────────────────────────────────────────────────────────


async def stream_song_chunks(
    storage_path: str,
) -> AsyncGenerator[bytes, None]:
    """Stream audio from Supabase Storage as CHUNK_SIZE byte chunks.

    Uses a signed URL so the service-role key is never exposed client-side.
    True streaming — does not buffer the entire file in memory.

    Args:
        storage_path: Path inside the bean-music bucket.

    Yields:
        bytes chunks of CHUNK_SIZE (last chunk may be smaller).

    Raises:
        RuntimeError: If the signed URL cannot be obtained.
        httpx.HTTPError: On network failure during streaming.
    """
    signed_url = await _get_signed_url(storage_path)

    logger.debug("Starting stream for path=%s", storage_path)
    chunk_count = 0
    total_bytes = 0

    try:
        async with httpx.AsyncClient(timeout=_STREAM_TIMEOUT) as http:
            async with http.stream("GET", signed_url) as response:
                response.raise_for_status()

                async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
                    if chunk:
                        yield chunk
                        chunk_count += 1
                        total_bytes += len(chunk)

    except httpx.HTTPStatusError as exc:
        logger.error(
            "HTTP error streaming song %s: %s %s",
            storage_path,
            exc.response.status_code,
            exc.response.text[:100],
        )
        raise
    except httpx.TimeoutException as exc:
        logger.error("Stream timed out for path=%s: %s", storage_path, exc)
        raise
    except Exception as exc:
        logger.error("Stream failed for path=%s: %s", storage_path, exc)
        raise

    logger.debug(
        "Stream complete: path=%s chunks=%d bytes=%d",
        storage_path,
        chunk_count,
        total_bytes,
    )


# ── Song management ───────────────────────────────────────────────────────────


async def register_uploaded_song(
    *,
    user_id: str,
    storage_path: str,
    title: str,
    mood: str,
    mime_type: str,
    file_size_bytes: int,
    duration_seconds: float | None = None,
) -> dict[str, Any]:
    """Insert a user_songs metadata row after the file has been uploaded to Storage.

    Returns the created row.
    Raises ValueError if mood is invalid or mime_type not allowed.
    """
    if mood not in VALID_MOODS:
        raise ValueError(f"Invalid mood {mood!r}. Must be one of: {sorted(VALID_MOODS)}")
    if mime_type not in ALLOWED_MIME_TYPES:
        raise ValueError(
            f"Unsupported file type {mime_type!r}. "
            f"Allowed: {sorted(ALLOWED_MIME_TYPES)}"
        )

    try:
        client = await get_service_client()
        result = await asyncio.wait_for(
            client.table("user_songs")
            .insert(
                {
                    "user_id": user_id,
                    "title": title.strip()[:200],
                    "mood": mood,
                    "storage_path": storage_path,
                    "mime_type": mime_type,
                    "file_size_bytes": file_size_bytes,
                    "duration_seconds": duration_seconds,
                    "is_default": False,
                }
            )
            .execute(),
            timeout=_DB_TIMEOUT,
        )

        row = (result.data or [{}])[0]
        logger.info(
            "Song registered: id=%s user=%s title=%r mood=%s",
            str(row.get("id", "?"))[:8],
            user_id[:8],
            title,
            mood,
        )
        return row

    except Exception as exc:
        logger.error("register_uploaded_song failed: %s", exc)
        raise


async def delete_user_song(song_id: str, user_id: str) -> bool:
    """Delete a user's song: remove from Storage and delete the DB row.

    Enforces ownership — users can only delete their own non-default songs.
    Returns True on success, False if not found or not owned.
    """
    try:
        client = await get_service_client()

        # Fetch to get storage_path and verify ownership
        result = await asyncio.wait_for(
            client.table("user_songs")
            .select("storage_path, is_default")
            .eq("id", song_id)
            .eq("user_id", user_id)
            .maybe_single()
            .execute(),
            timeout=_DB_TIMEOUT,
        )

        if not result or not result.data:
            logger.warning(
                "delete_user_song: song %s not found or not owned by user %s",
                song_id[:8],
                user_id[:8],
            )
            return False

        row = result.data
        if row.get("is_default"):
            logger.warning(
                "delete_user_song: attempt to delete default song %s", song_id[:8]
            )
            return False

        storage_path: str = row["storage_path"]

        # Delete from Storage first
        try:
            await asyncio.wait_for(
                client.storage.from_(STORAGE_BUCKET).remove([storage_path]),
                timeout=5.0,
            )
        except Exception as exc:
            logger.warning(
                "Failed to delete storage file %s (will still delete DB row): %s",
                storage_path,
                exc,
            )

        # Delete DB row
        await asyncio.wait_for(
            client.table("user_songs").delete().eq("id", song_id).execute(),
            timeout=_DB_TIMEOUT,
        )

        logger.info(
            "Song deleted: id=%s user=%s path=%s",
            song_id[:8],
            user_id[:8],
            storage_path,
        )
        return True

    except Exception as exc:
        logger.error("delete_user_song failed: %s", exc)
        return False