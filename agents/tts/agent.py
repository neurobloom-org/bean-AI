"""BEAN AI — TTS Agent: synthesize_speech FunctionTool.

ElevenLabs-backed TTS utility layer for BEAN AI.

This module is not an LLM agent. It provides:
  - synthesize_speech()         Full-response TTS, Supabase-cached, returned as base64.
                                Wrapped as an ADK FunctionTool for tool-based flows.
  - stream_tts_chunks()         True streaming TTS for low-latency robot playback.
                                The orchestrator calls this and forwards chunks to the WS.
  - get_cached_tts()            Direct cache read for pre-warmed filler audio.
  - get_cached_filler_audio()   Alias for get_cached_tts() (semantic clarity).
  - save_tts_cache()            Persist audio to the Supabase tts_cache table.

Voice ID resolution (same rule everywhere):
    Explicit voice_id argument → settings.elevenlabs_voice_id → ElevenLabsError.

Cache schema (tts_cache table):
    cache_key  TEXT  UNIQUE   SHA-256 of "v1:<voice_id>:<text>"
    phrase     TEXT           The source text (for human inspection only)
    voice_id   TEXT           The voice used to generate this audio
    audio_b64  TEXT           Base64-encoded raw audio bytes
    expires_at TIMESTAMPTZ    7-day TTL set by the DB default

Note: The cache column is named "phrase", not "text". Do not rename it here.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Final

from google.adk.tools import FunctionTool

from services.elevenlabs_service import synthesize_speech_full, synthesize_speech_stream
from services.supabase_client import get_service_client
from shared.config import get_settings
from shared.exceptions import ElevenLabsError
from shared.schemas import TTSChunk

__all__ = [
    "synthesize_speech",
    "stream_tts_chunks",
    "get_cached_tts",
    "get_cached_filler_audio",
    "save_tts_cache",
    "synthesize_speech_tool",
]

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────

_CACHE_TIMEOUT_SECONDS: Final[float] = 5.0
_FULL_TTS_TIMEOUT_SECONDS: Final[float] = 20.0
_MAX_TEXT_LENGTH: Final[int] = 10_000
_CACHE_KEY_VERSION: Final[str] = "v1"

# Strong references to fire-and-forget cache-save tasks.
# Without this, the GC can collect a task before it finishes.
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()

# Keys for cache saves that are currently in flight.
# Prevents duplicate concurrent saves for the same cache key.
_RESERVED_CACHE_SAVE_KEYS: set[str] = set()


# ── Validation helpers ────────────────────────────────────────────────────────


def _sanitize_for_log(value: str, max_length: int = 120) -> str:
    """Return a shortened log-safe representation of a string."""
    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}…"


def _normalize_text(text: Any) -> str:
    """Normalize and validate TTS input text.

    Raises:
        TypeError:  If text is not a string.
        ValueError: If text is empty or exceeds _MAX_TEXT_LENGTH.
    """
    if not isinstance(text, str):
        raise TypeError(
            f"Text for TTS must be a string, got {type(text).__name__} instead."
        )

    normalized = text.strip()
    if not normalized:
        raise ValueError("Text for TTS cannot be empty.")

    if len(normalized) > _MAX_TEXT_LENGTH:
        raise ValueError(
            f"Text for TTS exceeds maximum length of {_MAX_TEXT_LENGTH} characters "
            f"(got {len(normalized)})."
        )

    return normalized


def _resolve_voice_id(voice_id: str | None) -> str:
    """Return the effective ElevenLabs voice ID.

    Priority:
        1. voice_id argument (if non-empty after stripping)
        2. settings.elevenlabs_voice_id

    Raises:
        ElevenLabsError: If neither source provides a usable voice ID.
        TypeError:       If voice_id is provided but is not a string.
    """
    if voice_id is not None:
        if not isinstance(voice_id, str):
            raise TypeError(
                f"voice_id must be a string when provided, "
                f"got {type(voice_id).__name__}."
            )
        stripped = voice_id.strip()
        if stripped:
            return stripped

    settings = get_settings()
    configured = getattr(settings, "elevenlabs_voice_id", None)

    if isinstance(configured, str) and configured.strip():
        return configured.strip()

    raise ElevenLabsError(
        "No ElevenLabs voice ID available. Provide voice_id explicitly "
        "or set ELEVENLABS_VOICE_ID in your environment."
    )


def _build_tts_cache_key(text: str, voice_id: str) -> str:
    """Build a deterministic SHA-256 cache key for TTS output.

    Different voices for the same text produce different keys.
    """
    payload = f"{_CACHE_KEY_VERSION}:{voice_id}:{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_plausible_base64_string(value: Any) -> bool:
    """Return True if value looks like a non-empty base64 string."""
    return isinstance(value, str) and bool(value.strip())


def _validate_audio_bytes(value: Any) -> bytes:
    """Validate a full-audio TTS response payload.

    Raises:
        TypeError:  If value is not bytes-like.
        ValueError: If value is empty.
    """
    if not isinstance(value, (bytes, bytearray)):
        raise TypeError(
            "synthesize_speech_full() must return raw audio bytes, "
            f"got {type(value).__name__} instead."
        )

    normalized = bytes(value)
    if not normalized:
        raise ValueError("synthesize_speech_full() returned empty audio bytes.")

    return normalized


def _validate_tts_chunk(chunk: Any) -> TTSChunk:
    """Validate a TTSChunk yielded by synthesize_speech_stream().

    Raises:
        TypeError:  If chunk is not a TTSChunk, or audio_chunk is not bytes.
        ValueError: If a non-final chunk has empty audio_chunk bytes.
    """
    if not isinstance(chunk, TTSChunk):
        raise TypeError(
            "synthesize_speech_stream() must yield TTSChunk instances, "
            f"got {type(chunk).__name__} instead."
        )

    audio_chunk = chunk.audio_chunk
    is_final = chunk.is_final

    if not isinstance(audio_chunk, bytes):
        raise TypeError(
            f"TTSChunk.audio_chunk must be bytes, got {type(audio_chunk).__name__}."
        )

    # Final chunks may have empty audio (sentinel signal). Non-final must not.
    if not is_final and not audio_chunk:
        raise ValueError(
            "TTSChunk.audio_chunk cannot be empty for non-final chunks."
        )

    return chunk


def _build_success_response(
    *,
    audio_b64: str,
    voice_id: str,
    text_length: int,
    turn_id: str,
    cache_hit: bool,
    cache_key: str,
    audio_length_bytes: int | None,
) -> dict[str, Any]:
    """Build the standard success response dict for synthesize_speech()."""
    return {
        "status": "ok",
        "audio_b64": audio_b64,
        "audio_length_bytes": audio_length_bytes,
        "voice_id": voice_id,
        "text_length": text_length,
        "turn_id": turn_id,
        "cache_hit": cache_hit,
        "cache_key": cache_key,
    }


def _build_error_response(exc: Exception) -> dict[str, Any]:
    """Build the standard error response dict for synthesize_speech()."""
    return {
        "status": "error",
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


# ── Background task helpers ───────────────────────────────────────────────────


def _track_background_task(task: asyncio.Task[Any]) -> None:
    """Hold a strong reference to a background task until it completes."""
    _BACKGROUND_TASKS.add(task)

    def _on_done(done_task: asyncio.Task[Any]) -> None:
        _BACKGROUND_TASKS.discard(done_task)
        try:
            exc = done_task.exception()
        except asyncio.CancelledError:
            logger.debug("Background TTS cache-save task cancelled.")
            return

        if exc is not None:
            # exc_info=exc attaches the stored traceback — logger.exception()
            # would not work here because we are outside an exception handler.
            logger.error(
                "Background TTS cache-save task failed: %s",
                exc,
                exc_info=exc,
            )

    task.add_done_callback(_on_done)


def _schedule_background_cache_save(
    *,
    cache_key: str,
    audio_b64: str,
    text: str,
    voice_id: str,
) -> None:
    """Schedule a cache write without blocking the TTS response path.

    Skips silently if a save is already in flight for this cache_key to
    avoid duplicate concurrent writes to the same row.
    """
    if cache_key in _RESERVED_CACHE_SAVE_KEYS:
        logger.debug(
            "TTS cache: skipping duplicate in-flight save for key=%s…",
            cache_key[:12],
        )
        return

    _RESERVED_CACHE_SAVE_KEYS.add(cache_key)

    async def _runner() -> None:
        try:
            await save_tts_cache(
                cache_key=cache_key,
                audio_b64=audio_b64,
                text=text,
                voice_id=voice_id,
            )
        finally:
            _RESERVED_CACHE_SAVE_KEYS.discard(cache_key)

    try:
        task = asyncio.create_task(_runner(), name=f"tts-cache-save-{cache_key[:8]}")
    except RuntimeError:
        _RESERVED_CACHE_SAVE_KEYS.discard(cache_key)
        logger.warning(
            "TTS cache: no running event loop — skipping cache save for key=%s…",
            cache_key[:12],
        )
        return
    except Exception:
        _RESERVED_CACHE_SAVE_KEYS.discard(cache_key)
        logger.exception(
            "TTS cache: failed to schedule cache save for key=%s…",
            cache_key[:12],
        )
        return

    _track_background_task(task)


# ── Supabase cache helpers ────────────────────────────────────────────────────


async def get_cached_tts(cache_key: str) -> str | None:
    """Return base64-encoded audio from the Supabase tts_cache table.

    Returns None on cache miss, timeout, or any DB error.
    Never raises — cache misses must not break TTS.
    """
    if not isinstance(cache_key, str) or not cache_key.strip():
        return None

    try:
        client = await get_service_client()
        result = await asyncio.wait_for(
            client.table("tts_cache")
            .select("audio_b64")
            .eq("cache_key", cache_key)
            .maybe_single()
            .execute(),
            timeout=_CACHE_TIMEOUT_SECONDS,
        )

        if result is None or not result.data:
            return None

        data = result.data
        if not isinstance(data, dict):
            return None

        audio_b64 = data.get("audio_b64")
        if not isinstance(audio_b64, str) or not audio_b64.strip():
            logger.warning(
                "TTS cache: invalid cached audio payload for key=%s…",
                cache_key[:12],
            )
            return None

        return audio_b64

    except asyncio.TimeoutError:
        logger.warning(
            "TTS cache: lookup timed out after %.1fs for key=%s…",
            _CACHE_TIMEOUT_SECONDS,
            cache_key[:12],
        )
        return None

    except Exception as exc:
        logger.warning(
            "TTS cache: lookup failed for key=%s…: %s",
            cache_key[:12],
            exc,
        )
        return None


async def save_tts_cache(
    *,
    cache_key: str,
    audio_b64: str,
    text: str,
    voice_id: str,
) -> None:
    """Persist synthesized audio to the Supabase tts_cache table.

    Best-effort: failures are logged but never re-raised.
    The cache column for source text is named "phrase" (not "text") —
    this matches the 001_schema.sql definition exactly.
    """
    if not cache_key or not _is_plausible_base64_string(audio_b64):
        logger.warning("TTS cache: skipping save — invalid cache_key or audio_b64.")
        return

    try:
        client = await get_service_client()
        await asyncio.wait_for(
            client.table("tts_cache")
            .upsert(
                {
                    "cache_key": cache_key,
                    "phrase": text,        # column is "phrase" per 001_schema.sql
                    "voice_id": voice_id,
                    "audio_b64": audio_b64,
                },
                on_conflict="cache_key",
            )
            .execute(),
            timeout=_CACHE_TIMEOUT_SECONDS,
        )
        logger.debug(
            "TTS cache: saved key=%s… voice=%s",
            cache_key[:12],
            voice_id[:8],
        )

    except asyncio.TimeoutError:
        logger.warning(
            "TTS cache: save timed out after %.1fs for key=%s…",
            _CACHE_TIMEOUT_SECONDS,
            cache_key[:12],
        )

    except Exception as exc:
        logger.warning(
            "TTS cache: save failed for key=%s…: %s",
            cache_key[:12],
            exc,
        )


async def get_cached_filler_audio(cache_key: str) -> str | None:
    """Return pre-cached filler phrase audio from the Supabase tts_cache table.

    Semantic alias for get_cached_tts() — makes call sites more readable.
    """
    return await get_cached_tts(cache_key)


# ── Full-audio path ───────────────────────────────────────────────────────────


async def _call_synthesize_speech_full(
    *,
    text: str,
    turn_id: str,
    voice_id: str,
) -> bytes:
    """Call the ElevenLabs full-audio service and validate the result."""
    try:
        raw_audio = await asyncio.wait_for(
            synthesize_speech_full(text, turn_id=turn_id, voice_id=voice_id),
            timeout=_FULL_TTS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            f"Full-audio TTS request timed out after {_FULL_TTS_TIMEOUT_SECONDS}s."
        ) from exc

    return _validate_audio_bytes(raw_audio)


async def synthesize_speech(
    text: str,
    voice_id: str | None = None,
) -> dict[str, Any]:
    """Synthesize full speech audio from text and return it as base64.

    Checks the Supabase tts_cache before calling ElevenLabs. On a cache
    miss, schedules a background write so future calls return instantly.

    For low-latency robot playback, use stream_tts_chunks() instead —
    this function waits for the complete audio before returning.

    Args:
        text:     Text to synthesize.
        voice_id: ElevenLabs voice ID override. Falls back to settings.

    Returns:
        Dict with keys: status, audio_b64, audio_length_bytes, voice_id,
        text_length, turn_id, cache_hit, cache_key.
        On error: {status: "error", error: str, error_type: str}.
    """
    turn_id = str(uuid.uuid4())

    try:
        normalized_text = _normalize_text(text)
        effective_voice_id = _resolve_voice_id(voice_id)
        cache_key = _build_tts_cache_key(normalized_text, effective_voice_id)

        cached_audio_b64 = await get_cached_tts(cache_key)
        if cached_audio_b64 is not None:
            logger.debug(
                "TTS cache HIT turn=%s key=%s…", turn_id[:8], cache_key[:12]
            )
            return _build_success_response(
                audio_b64=cached_audio_b64,
                voice_id=effective_voice_id,
                text_length=len(normalized_text),
                turn_id=turn_id,
                cache_hit=True,
                cache_key=cache_key,
                audio_length_bytes=None,
            )

        audio_bytes = await _call_synthesize_speech_full(
            text=normalized_text,
            turn_id=turn_id,
            voice_id=effective_voice_id,
        )
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        _schedule_background_cache_save(
            cache_key=cache_key,
            audio_b64=audio_b64,
            text=normalized_text,
            voice_id=effective_voice_id,
        )

        return _build_success_response(
            audio_b64=audio_b64,
            voice_id=effective_voice_id,
            text_length=len(normalized_text),
            turn_id=turn_id,
            cache_hit=False,
            cache_key=cache_key,
            audio_length_bytes=len(audio_bytes),
        )

    except Exception as exc:
        logger.exception("synthesize_speech failed: turn_id=%s", turn_id)
        return _build_error_response(exc)


# ── Streaming path ────────────────────────────────────────────────────────────


async def _iterate_synthesize_speech_stream(
    *,
    text: str,
    turn_id: str,
    voice_id: str,
) -> AsyncGenerator[TTSChunk, None]:
    """Iterate the ElevenLabs stream and validate each TTSChunk.

    Applies a tight timeout to the first chunk (startup latency) and a
    per-chunk timeout to subsequent chunks (stall detection). Both timeouts
    are enforced inside synthesize_speech_stream() in the service layer.
    """
    stream_obj = synthesize_speech_stream(
        text,
        turn_id=turn_id,
        voice_id=voice_id,
    )

    # synthesize_speech_stream is an async generator function.
    # We need to call it to get the async generator, then iterate.
    try:
        iterator: AsyncIterator[TTSChunk] = stream_obj.__aiter__()
    except AttributeError as exc:
        raise TypeError(
            "synthesize_speech_stream() must return an async iterable."
        ) from exc

    async for chunk in iterator:
        yield _validate_tts_chunk(chunk)


async def stream_tts_chunks(
    text: str,
    voice_id: str | None = None,
) -> AsyncGenerator[TTSChunk, None]:
    """Stream TTS audio chunks for WebSocket delivery.

    Yields TTSChunk objects as audio arrives from ElevenLabs — true streaming,
    not buffered. The orchestrator iterates this and sends each chunk as a
    binary WebSocket frame.

    This path is intentionally not cached. Streaming is used for real-time
    robot playback; the cache is used for pre-generated filler audio and the
    full-audio (synthesize_speech) path.

    Args:
        text:     Text to synthesize.
        voice_id: ElevenLabs voice ID override. Falls back to settings.

    Yields:
        TTSChunk instances. The last chunk has is_final=True.

    Raises:
        TypeError:       If text is not a string.
        ValueError:      If text is empty or exceeds _MAX_TEXT_LENGTH.
        ElevenLabsError: If the stream fails or stalls.
    """
    normalized_text = _normalize_text(text)
    effective_voice_id = _resolve_voice_id(voice_id)
    turn_id = str(uuid.uuid4())

    logger.debug(
        "TTS stream start: turn=%s text_len=%d voice=%s text=%r",
        turn_id[:8],
        len(normalized_text),
        effective_voice_id[:8],
        _sanitize_for_log(normalized_text),
    )

    try:
        async for chunk in _iterate_synthesize_speech_stream(
            text=normalized_text,
            turn_id=turn_id,
            voice_id=effective_voice_id,
        ):
            yield chunk

    except Exception:
        logger.exception("stream_tts_chunks failed: turn_id=%s", turn_id)
        raise


# ── ADK FunctionTool registration ─────────────────────────────────────────────

synthesize_speech_tool = FunctionTool(func=synthesize_speech)
