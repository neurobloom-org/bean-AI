"""BEAN AI — TTS Agent: synthesize_speech FunctionTool.

ElevenLabs-backed TTS utility layer for BEAN AI.

This module is not an LLM agent. It provides:
- a FunctionTool wrapper for full-response TTS
- a direct async generator for streaming TTS chunks to the orchestrator
- Supabase-backed cache helpers

The orchestrator should use `stream_tts_chunks()` for real-time robot playback.
The ADK tool `synthesize_speech_tool` is intended for simpler full-audio use cases.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from google.adk.tools import FunctionTool

from services.elevenlabs_service import synthesize_speech_full, synthesize_speech_stream
from services.supabase_client import get_service_client
from shared.config import get_settings
from shared.schemas import TTSChunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

_CACHE_TIMEOUT_SECONDS = 5.0
_FULL_TTS_TIMEOUT_SECONDS = 20.0
_STREAM_FIRST_CHUNK_TIMEOUT_SECONDS = 10.0
_MAX_TEXT_LENGTH = 10_000
_CACHE_KEY_VERSION = "v1"

# Keep strong refs to fire-and-forget tasks until they finish.
_BACKGROUND_TASKS: set[asyncio.Task[Any]] = set()

# Reserved keys for in-flight background cache saves.
_RESERVED_CACHE_SAVE_KEYS: set[str] = set()

# ---------------------------------------------------------------------------
# Service capability detection — evaluate once at module load
# ---------------------------------------------------------------------------

try:
    _FULL_SUPPORTS_VOICE_ID = (
        "voice_id" in inspect.signature(synthesize_speech_full).parameters
    )
except (TypeError, ValueError):
    _FULL_SUPPORTS_VOICE_ID = False

try:
    _STREAM_SUPPORTS_VOICE_ID = (
        "voice_id" in inspect.signature(synthesize_speech_stream).parameters
    )
except (TypeError, ValueError):
    _STREAM_SUPPORTS_VOICE_ID = False


# ---------------------------------------------------------------------------
# Validation and normalization helpers
# ---------------------------------------------------------------------------


def _sanitize_for_log(value: str, max_length: int = 120) -> str:
    """Return a shortened log-safe representation."""
    if len(value) <= max_length:
        return value
    return f"{value[:max_length]}…"


def _normalize_text(text: Any) -> str:
    """Normalize and validate TTS input text."""
    if not isinstance(text, str):
        raise TypeError(
            f"Text for TTS must be a string, got {type(text).__name__} instead."
        )

    normalized = text.strip()
    if not normalized:
        raise ValueError("Text for TTS cannot be empty.")

    if len(normalized) > _MAX_TEXT_LENGTH:
        raise ValueError(
            f"Text for TTS exceeds maximum length of {_MAX_TEXT_LENGTH} characters."
        )

    return normalized


def _normalize_voice_id(value: Any) -> str | None:
    """Normalize a voice ID value if possible."""
    if value is None:
        return None

    if not isinstance(value, str):
        raise TypeError(
            f"voice_id must be a string when provided, got {type(value).__name__}."
        )

    normalized = value.strip()
    return normalized or None


def _resolve_voice_id(voice_id: str | None) -> str:
    """Resolve the effective ElevenLabs voice ID."""
    explicit_voice_id = _normalize_voice_id(voice_id)
    if explicit_voice_id:
        return explicit_voice_id

    settings = get_settings()
    settings_voice_id = _normalize_voice_id(
        getattr(settings, "elevenlabs_voice_id", None)
    )

    if not settings_voice_id:
        raise ValueError(
            "No ElevenLabs voice ID configured. Provide voice_id explicitly "
            "or set settings.elevenlabs_voice_id."
        )

    return settings_voice_id


def _build_tts_cache_key(text: str, voice_id: str) -> str:
    """Build a deterministic cache key for TTS output."""
    payload = f"{_CACHE_KEY_VERSION}:{voice_id}:{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_plausible_base64_string(value: Any) -> bool:
    """Cheap validation for cached base64 payloads."""
    return isinstance(value, str) and bool(value.strip())


def _validate_audio_bytes(value: Any) -> bytes:
    """Validate a full-audio TTS response."""
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
    """Validate a streamed TTS chunk."""
    if not isinstance(chunk, TTSChunk):
        raise TypeError(
            "synthesize_speech_stream() must yield TTSChunk instances, "
            f"got {type(chunk).__name__} instead."
        )

    audio_chunk = getattr(chunk, "audio_chunk", None)
    is_final = bool(getattr(chunk, "is_final", False))

    if audio_chunk is not None and not isinstance(audio_chunk, bytes):
        raise TypeError("TTSChunk.audio_chunk must be bytes when present.")

    if not is_final and audio_chunk is not None and not audio_chunk:
        raise ValueError("TTSChunk.audio_chunk cannot be empty for non-final chunks.")

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
    """Build a consistent success response payload."""
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
    """Build a consistent error response payload."""
    return {
        "status": "error",
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


# ---------------------------------------------------------------------------
# Background task helpers
# ---------------------------------------------------------------------------


def _track_background_task(task: asyncio.Task[Any]) -> None:
    """Track a background task until completion."""
    _BACKGROUND_TASKS.add(task)

    def _cleanup(done_task: asyncio.Task[Any]) -> None:
        _BACKGROUND_TASKS.discard(done_task)
        try:
            exc = done_task.exception()
        except asyncio.CancelledError:
            logger.debug("Background TTS task was cancelled.")
            return

        if exc is not None:
            logger.exception("Background TTS task failed: %s", exc)

    task.add_done_callback(_cleanup)


def _schedule_background_cache_save(
    *,
    cache_key: str,
    audio_b64: str,
    text: str,
    voice_id: str,
) -> None:
    """Schedule cache persistence without blocking the TTS response path."""
    if cache_key in _RESERVED_CACHE_SAVE_KEYS:
        logger.debug("Skipping duplicate pending cache save for key=%s", cache_key)
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
        task = asyncio.create_task(_runner())
    except RuntimeError:
        _RESERVED_CACHE_SAVE_KEYS.discard(cache_key)
        logger.warning(
            "No running event loop; skipping background TTS cache save for key=%s",
            cache_key,
        )
        return
    except Exception:
        _RESERVED_CACHE_SAVE_KEYS.discard(cache_key)
        logger.exception(
            "Failed to schedule background TTS cache save for key=%s",
            cache_key,
        )
        return

    _track_background_task(task)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


async def get_cached_tts(cache_key: str) -> str | None:
    """Return base64-encoded audio from the Supabase TTS cache."""
    if not isinstance(cache_key, str) or not cache_key.strip():
        return None

    try:
        client = await get_service_client()
        # AFTER — proper fix
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

        # isinstance check lets mypy narrow the type to str — no cast needed
        if not isinstance(audio_b64, str) or not audio_b64.strip():
            logger.warning(
                "Ignoring invalid cached audio payload for key=%s", cache_key
            )
            return None

        return audio_b64  # mypy now knows this is str, not Any
    except asyncio.TimeoutError:
        logger.warning("TTS cache lookup timed out for key=%s", cache_key)
        return None
    except Exception as exc:
        logger.warning("TTS cache lookup failed for key=%s: %s", cache_key, exc)
        return None


async def save_tts_cache(
    *,
    cache_key: str,
    audio_b64: str,
    text: str,
    voice_id: str,
) -> None:
    """Persist synthesized audio to the Supabase TTS cache.

    This function is best-effort only. Cache write failures must never break TTS.
    """
    if not cache_key or not _is_plausible_base64_string(audio_b64):
        logger.warning("Skipping TTS cache save due to invalid payload.")
        return

    try:
        client = await get_service_client()
        payload = {
            "cache_key": cache_key,
            "audio_b64": audio_b64,
            "text": text,
            "voice_id": voice_id,
        }

        await asyncio.wait_for(
            client.table("tts_cache").upsert(payload).execute(),
            timeout=_CACHE_TIMEOUT_SECONDS,
        )

    except asyncio.TimeoutError:
        logger.warning("TTS cache save timed out for key=%s", cache_key)
    except Exception as exc:
        logger.warning("TTS cache save failed for key=%s: %s", cache_key, exc)


async def get_cached_filler_audio(cache_key: str) -> str | None:
    """Return pre-cached filler phrase audio from the Supabase TTS cache."""
    return await get_cached_tts(cache_key)


# ---------------------------------------------------------------------------
# ElevenLabs full-audio helpers
# ---------------------------------------------------------------------------


async def _call_synthesize_speech_full(
    *,
    text: str,
    turn_id: str,
    voice_id: str,
) -> bytes:
    """Call full-audio TTS service with compatibility handling."""
    kwargs: dict[str, Any] = {"turn_id": turn_id}

    if _FULL_SUPPORTS_VOICE_ID:
        kwargs["voice_id"] = voice_id

    try:
        raw_audio = await asyncio.wait_for(
            synthesize_speech_full(text, **kwargs),
            timeout=_FULL_TTS_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise TimeoutError("Full-audio TTS request timed out.") from exc

    return _validate_audio_bytes(raw_audio)


# ---------------------------------------------------------------------------
# Public full-audio API
# ---------------------------------------------------------------------------


async def synthesize_speech(
    text: str,
    voice_id: str | None = None,
) -> dict[str, Any]:
    """Synthesize full speech audio from text.

    This returns the complete audio as base64 for simpler tool-based use cases.
    For low-latency robot playback, use `stream_tts_chunks()` instead.

    Cache behavior:
    - checks Supabase cache before calling ElevenLabs
    - schedules generated audio to be written back after a miss
    """
    turn_id = str(uuid.uuid4())

    try:
        normalized_text = _normalize_text(text)
        effective_voice_id = _resolve_voice_id(voice_id)
        cache_key = _build_tts_cache_key(normalized_text, effective_voice_id)

        cached_audio_b64 = await get_cached_tts(cache_key)
        if cached_audio_b64 is not None:
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


# ---------------------------------------------------------------------------
# ElevenLabs streaming helpers
# ---------------------------------------------------------------------------


def _build_stream_kwargs(*, turn_id: str, voice_id: str) -> dict[str, Any]:
    """Build kwargs for streaming service invocation."""
    kwargs: dict[str, Any] = {"turn_id": turn_id}
    if _STREAM_SUPPORTS_VOICE_ID:
        kwargs["voice_id"] = voice_id
    return kwargs


async def _open_stream_iterator(
    *,
    text: str,
    turn_id: str,
    voice_id: str,
) -> AsyncIterator[TTSChunk]:
    """Create and validate the upstream stream iterator."""
    stream_obj = synthesize_speech_stream(
        text,
        **_build_stream_kwargs(turn_id=turn_id, voice_id=voice_id),
    )

    try:
        iterator = aiter(stream_obj)
    except TypeError as exc:
        raise TypeError(
            "synthesize_speech_stream() must return an async iterable."
        ) from exc

    return iterator


async def _iterate_synthesize_speech_stream(
    *,
    text: str,
    turn_id: str,
    voice_id: str,
) -> AsyncGenerator[TTSChunk, None]:
    """Stream TTS chunks with compatibility handling and validation.

    Applies a timeout only to the first chunk so the robot does not wait
    indefinitely for TTS startup.
    """
    iterator = await _open_stream_iterator(
        text=text,
        turn_id=turn_id,
        voice_id=voice_id,
    )

    try:
        first_chunk = await asyncio.wait_for(
            anext(iterator),
            timeout=_STREAM_FIRST_CHUNK_TIMEOUT_SECONDS,
        )
    except StopAsyncIteration:
        raise ValueError("synthesize_speech_stream() yielded no chunks.") from None
    except asyncio.TimeoutError as exc:
        raise TimeoutError(
            "Streaming TTS did not produce the first chunk in time."
        ) from exc

    yield _validate_tts_chunk(first_chunk)

    async for chunk in iterator:
        yield _validate_tts_chunk(chunk)


# ---------------------------------------------------------------------------
# Public streaming API
# ---------------------------------------------------------------------------


async def stream_tts_chunks(
    text: str,
    voice_id: str | None = None,
) -> AsyncGenerator[TTSChunk, None]:
    """Stream TTS audio chunks for WebSocket delivery.

    This path is intentionally uncached by default. Streaming is usually used
    for low-latency robot playback, and cache checks are better handled by the
    orchestrator if needed.
    """
    normalized_text = _normalize_text(text)
    effective_voice_id = _resolve_voice_id(voice_id)
    turn_id = str(uuid.uuid4())

    logger.debug(
        "Starting TTS stream: turn_id=%s text_length=%d voice_id=%s text=%r",
        turn_id,
        len(normalized_text),
        effective_voice_id,
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


synthesize_speech_tool = FunctionTool(func=synthesize_speech)
