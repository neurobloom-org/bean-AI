"""BEAN AI — Emotion Agent: detect_emotion FunctionTool.

Real-time wav2vec2 emotion inference on PCM audio windows. No LLM needed.
Returns emotion label + confidence to session state.

Production-oriented version:
- Preloadable model via preload_emotion_model()
- Inference runs off the async event loop via asyncio.to_thread()
- Temporal smoothing with EMA per session
- Neutral-biased EMA prior to avoid cold-start spikes
- Dynamic noise-floor gate per session
- Buffered batch logging to Supabase
- Background log worker health tracking + structured shutdown
- Auto-restartable background log worker
- Per-session state cleanup with TTL
- Zero-copy-ish PCM decode with numpy.frombuffer
- Strict sample-rate validation for model compatibility
- Lazy-initialized asyncio locks for safer import/runtime behavior

Inputs:
    audio_b64: base64-encoded PCM int16 mono audio
    session_id: current session ID for DB logging

Outputs:
    dict { label, confidence, timestamp, window_ms }
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from google.adk.tools import FunctionTool

from services.supabase_client import get_service_client
from shared.config import get_settings
from shared.enums import EmotionLabel
from shared.schemas import EmotionResult

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE = 16000
MODEL_EXPECTED_SAMPLE_RATE = 16000
MIN_AUDIO_MS = 100

# EMA smoothing factor: lower = smoother / slower changing mood
EMA_ALPHA = 0.30

# Neutral-biased Bayesian prior for session startup
NEUTRAL_PRIOR_WEIGHT = 0.80

# Dynamic noise floor
NOISE_FLOOR_BOOTSTRAP_WINDOWS = 6
NOISE_GATE_MULTIPLIER = 2.0
MIN_GATE_RMS = 0.003

# DB logging
LOG_EVERY_N_WINDOWS = 3
LOG_BATCH_SIZE = 50
LOG_FLUSH_INTERVAL_SECONDS = 10.0
MAX_LOG_BUFFER_SIZE = 5000

# Session state cleanup
SESSION_STATE_TTL_SECONDS = 30 * 60  # 30 minutes
SESSION_CLEANUP_INTERVAL = 200

# Worker health
WORKER_STALE_AFTER_SECONDS = LOG_FLUSH_INTERVAL_SECONDS * 3

_EMOTION_ORDER = [
    "angry",
    "calm",
    "disgusted",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]

# Plain string fallback so the module does not depend on EmotionLabel
# for fallback behavior during exception handling.
FALLBACK_LABEL = "neutral"

# ── Model state ──────────────────────────────────────────────────────────────
_model = None
_feature_extractor = None
_model_lock: asyncio.Lock = asyncio.Lock()

# ── Background logging worker state ──────────────────────────────────────────
_log_buffer: deque[dict[str, Any]] = deque()
_log_buffer_lock: asyncio.Lock = asyncio.Lock()

_log_worker_task: asyncio.Task[Any] | None = None
_log_worker_started = False
_log_worker_start_lock: asyncio.Lock = asyncio.Lock()
_log_worker_stop_event: asyncio.Event | None = None

_log_worker_last_heartbeat_ts: float = 0.0
_log_worker_last_error: str | None = None


# ── Per-session state ────────────────────────────────────────────────────────
@dataclass
class _SessionState:
    log_counter: int = 0
    last_seen_ts: float = 0.0
    noise_rms_samples: deque[float] = field(
        default_factory=lambda: deque(maxlen=NOISE_FLOOR_BOOTSTRAP_WINDOWS)
    )
    ema_probs: np.ndarray | None = None


_session_states: dict[str, _SessionState] = {}
_session_lock: asyncio.Lock = asyncio.Lock()
_cleanup_tick = 0


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _neutral_prior_probs() -> np.ndarray:
    """Return a neutral-biased starting distribution."""
    n = len(_EMOTION_ORDER)
    probs = np.full(n, (1.0 - NEUTRAL_PRIOR_WEIGHT) / (n - 1), dtype=np.float32)
    neutral_idx = _EMOTION_ORDER.index("neutral")
    probs[neutral_idx] = NEUTRAL_PRIOR_WEIGHT
    return probs


async def _load_model() -> None:
    """Lazy-load the wav2vec2 emotion recognition model once."""
    global _model, _feature_extractor

    if _model is not None and _feature_extractor is not None:
        return

    async with _model_lock:
        if _model is not None and _feature_extractor is not None:
            return

        try:
            from transformers import (
                AutoFeatureExtractor,
                AutoModelForAudioClassification,
            )

            settings = get_settings()
            model_name = settings.emotion_model_name

            logger.info("Loading emotion model: %s", model_name)
            _feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
            _model = AutoModelForAudioClassification.from_pretrained(model_name)
            _model.eval()
            logger.info("Emotion model loaded successfully")
        except Exception:
            logger.exception("Failed to load emotion model")
            raise


async def preload_emotion_model() -> None:
    """Call this during app startup."""
    await _load_model()


def _pcm_b64_to_float_array(audio_b64: str) -> np.ndarray:
    """Convert base64 PCM int16 audio to float32 numpy array.

    Uses numpy.frombuffer to avoid struct.unpack tuple allocation.
    """
    try:
        raw_bytes = base64.b64decode(audio_b64, validate=True)
    except Exception as exc:
        raise ValueError("Invalid base64 audio payload") from exc

    if not raw_bytes:
        return np.array([], dtype=np.float32)

    # PCM int16 needs 2-byte alignment.
    if len(raw_bytes) % 2 != 0:
        raw_bytes = raw_bytes[:-1]

    if not raw_bytes:
        return np.array([], dtype=np.float32)

    int16_array = np.frombuffer(raw_bytes, dtype=np.int16)
    return int16_array.astype(np.float32) / 32768.0


def _compute_window_ms(audio_array: np.ndarray, sample_rate: int) -> int:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")
    return int(len(audio_array) / sample_rate * 1000)


def _compute_rms(audio_array: np.ndarray) -> float:
    if len(audio_array) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_array))))


def _argmax_label_from_probs(probs: np.ndarray) -> str:
    idx = int(np.argmax(probs))
    return _EMOTION_ORDER[idx]


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float32)
    total = float(probs.sum())
    if total <= 0:
        return _neutral_prior_probs()
    return probs / total


def _run_inference_sync(audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
    """Run feature extraction + model inference synchronously.

    Execute this via asyncio.to_thread() so it does not block the event loop.
    Returns the raw softmax probabilities.
    """
    if _feature_extractor is None or _model is None:
        raise RuntimeError("Emotion model not loaded")

    import torch

    inputs = _feature_extractor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )

    with torch.inference_mode():
        logits = _model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

    return probs[0].detach().cpu().numpy().astype(np.float32)


def _cleanup_expired_sessions(now_ts: float) -> None:
    expired = [
        session_id
        for session_id, state in _session_states.items()
        if now_ts - state.last_seen_ts > SESSION_STATE_TTL_SECONDS
    ]
    for session_id in expired:
        _session_states.pop(session_id, None)


async def cleanup_emotion_session(session_id: str) -> None:
    """Optionally call when a session ends."""
    if not session_id:
        return

    async with _session_lock:
        _session_states.pop(session_id, None)


async def _get_or_create_session_state(session_id: str) -> _SessionState | None:
    global _cleanup_tick

    if not session_id:
        return None

    loop = asyncio.get_running_loop()
    now_ts = loop.time()

    async with _session_lock:
        state = _session_states.get(session_id)
        if state is None:
            state = _SessionState(
                last_seen_ts=now_ts,
                ema_probs=_neutral_prior_probs(),
            )
            _session_states[session_id] = state
        else:
            state.last_seen_ts = now_ts

        _cleanup_tick += 1
        if _cleanup_tick % SESSION_CLEANUP_INTERVAL == 0:
            _cleanup_expired_sessions(now_ts)

        return state


def _compute_dynamic_gate(state: _SessionState | None) -> float:
    if state is None or not state.noise_rms_samples:
        return MIN_GATE_RMS

    baseline = float(np.mean(state.noise_rms_samples))
    return max(MIN_GATE_RMS, baseline * NOISE_GATE_MULTIPLIER)


def _update_noise_floor(state: _SessionState | None, rms: float) -> None:
    if state is None:
        return
    state.noise_rms_samples.append(rms)


def _apply_ema(
    state: _SessionState | None,
    raw_probs: np.ndarray,
    alpha: float = EMA_ALPHA,
) -> np.ndarray:
    raw_probs = _normalize_probs(raw_probs)

    if state is None:
        return raw_probs

    if state.ema_probs is None:
        state.ema_probs = _neutral_prior_probs()

    state.ema_probs = _normalize_probs(
        alpha * raw_probs + (1.0 - alpha) * state.ema_probs
    )
    return state.ema_probs


def _should_log_this_window(state: _SessionState | None) -> bool:
    if state is None:
        return False
    state.log_counter += 1
    return state.log_counter % LOG_EVERY_N_WINDOWS == 0


def _get_loop_time() -> float:
    try:
        return asyncio.get_running_loop().time()
    except RuntimeError:
        return 0.0


def get_emotion_log_worker_health() -> dict[str, Any]:
    """Health snapshot for monitoring / orchestrator checks."""
    now_ts = _get_loop_time()
    stale = False
    if _log_worker_last_heartbeat_ts > 0 and now_ts > 0:
        stale = (now_ts - _log_worker_last_heartbeat_ts) > WORKER_STALE_AFTER_SECONDS

    task_done = _log_worker_task.done() if _log_worker_task is not None else False

    return {
        "started": _log_worker_started,
        "running": _log_worker_task is not None and not task_done,
        "task_done": task_done,
        "buffer_size": len(_log_buffer),
        "last_heartbeat_ts": _log_worker_last_heartbeat_ts,
        "stale": stale,
        "last_error": _log_worker_last_error,
    }


def _log_worker_done(task: asyncio.Task[Any]) -> None:
    """Capture worker failure so it never dies silently and can restart later."""
    global _log_worker_last_error, _log_worker_started, _log_worker_task

    try:
        task.result()
    except asyncio.CancelledError:
        logger.info("Emotion log worker cancelled")
    except Exception as exc:
        _log_worker_last_error = str(exc)
        logger.exception(
            "Emotion log worker crashed — will restart on next detect_emotion call"
        )
    finally:
        _log_worker_started = False
        _log_worker_task = None


async def _ensure_log_worker_started() -> None:
    global \
        _log_worker_task, \
        _log_worker_started, \
        _log_worker_stop_event, \
        _log_worker_last_error

    health = get_emotion_log_worker_health()
    if health["running"]:
        return

    async with _log_worker_start_lock:
        health = get_emotion_log_worker_health()
        if health["running"]:
            return

        _log_worker_stop_event = asyncio.Event()
        _log_worker_task = asyncio.create_task(
            _log_worker_loop(), name="emotion-log-worker"
        )
        _log_worker_task.add_done_callback(_log_worker_done)
        _log_worker_started = True
        _log_worker_last_error = None
        logger.info("Emotion log worker started")


async def _enqueue_emotion_log(row: dict[str, Any]) -> None:
    await _ensure_log_worker_started()

    async with _log_buffer_lock:
        if len(_log_buffer) >= MAX_LOG_BUFFER_SIZE:
            dropped = _log_buffer.popleft()
            logger.warning(
                "Emotion log buffer full; dropping oldest row for session=%s",
                dropped.get("session_id", ""),
            )
        _log_buffer.append(row)


async def _drain_log_buffer(max_items: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    async with _log_buffer_lock:
        while _log_buffer and len(rows) < max_items:
            rows.append(_log_buffer.popleft())

    return rows


async def _flush_log_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    try:
        client = await get_service_client()
        await client.table("emotion_events").insert(rows).execute()
        logger.debug("Flushed %d emotion logs", len(rows))
    except Exception:
        logger.exception("Failed to flush emotion logs")

        requeued = 0
        dropped = 0

        async with _log_buffer_lock:
            for row in reversed(rows):
                if len(_log_buffer) < MAX_LOG_BUFFER_SIZE:
                    _log_buffer.appendleft(row)
                    requeued += 1
                else:
                    dropped += 1

        if dropped:
            logger.warning(
                "Dropped %d emotion log rows after flush failure; requeued=%d",
                dropped,
                requeued,
            )


async def _log_worker_loop() -> None:
    global _log_worker_last_heartbeat_ts, _log_worker_last_error

    _log_worker_last_error = None

    if _log_worker_stop_event is None:
        raise RuntimeError("Log worker stop event not initialized")

    while not _log_worker_stop_event.is_set():
        _log_worker_last_heartbeat_ts = _get_loop_time()

        try:
            await asyncio.wait_for(
                _log_worker_stop_event.wait(),
                timeout=LOG_FLUSH_INTERVAL_SECONDS,
            )
        except asyncio.TimeoutError:
            pass

        rows = await _drain_log_buffer(LOG_BATCH_SIZE)
        while rows:
            await _flush_log_rows(rows)
            _log_worker_last_heartbeat_ts = _get_loop_time()
            rows = await _drain_log_buffer(LOG_BATCH_SIZE)

    # Final flush during structured shutdown
    rows = await _drain_log_buffer(LOG_BATCH_SIZE)
    while rows:
        await _flush_log_rows(rows)
        rows = await _drain_log_buffer(LOG_BATCH_SIZE)


async def flush_emotion_logs_now() -> None:
    """Flush any buffered logs immediately."""
    rows = await _drain_log_buffer(LOG_BATCH_SIZE)
    while rows:
        await _flush_log_rows(rows)
        rows = await _drain_log_buffer(LOG_BATCH_SIZE)


async def shutdown_emotion_logging() -> None:
    """Structured shutdown hook for SIGTERM / app shutdown."""
    global _log_worker_task, _log_worker_stop_event

    if _log_worker_stop_event is not None:
        _log_worker_stop_event.set()

    if _log_worker_task is not None:
        try:
            await _log_worker_task
        except asyncio.CancelledError:
            pass

    await flush_emotion_logs_now()


def _build_low_signal_response(
    *,
    timestamp: str,
    window_ms: int,
    status: str,
) -> dict[str, Any]:
    return {
        "label": FALLBACK_LABEL,
        "confidence": 0.0,
        "timestamp": timestamp,
        "window_ms": window_ms,
        "status": status,
    }


async def detect_emotion(
    audio_b64: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    session_id: str = "",
    user_id: str = "",
) -> dict[str, Any]:
    """Detect emotion from a PCM audio window.

    Args:
        audio_b64: Base64-encoded PCM int16 mono audio.
        sample_rate: Audio sample rate in Hz.
        session_id: Current session ID for DB logging.

    Returns:
        Dict with keys: label, confidence, timestamp, window_ms.
    """
    timestamp = _utc_now_iso()

    try:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")

        if sample_rate != MODEL_EXPECTED_SAMPLE_RATE:
            logger.error(
                "Emotion agent received sample_rate=%d but model expects %d Hz",
                sample_rate,
                MODEL_EXPECTED_SAMPLE_RATE,
            )
            return {
                "label": FALLBACK_LABEL,
                "confidence": 0.0,
                "timestamp": timestamp,
                "window_ms": 0,
                "status": "audio_wrong_sample_rate",
            }

        await _load_model()

        audio_array = _pcm_b64_to_float_array(audio_b64)
        min_samples = max(1, sample_rate * MIN_AUDIO_MS // 1000)

        if len(audio_array) < min_samples:
            return _build_low_signal_response(
                timestamp=timestamp,
                window_ms=0,
                status="audio_too_short",
            )

        window_ms = _compute_window_ms(audio_array, sample_rate)
        rms = _compute_rms(audio_array)

        state = await _get_or_create_session_state(session_id)

        _update_noise_floor(state, rms)
        dynamic_gate = _compute_dynamic_gate(state)

        if rms < dynamic_gate:
            return _build_low_signal_response(
                timestamp=timestamp,
                window_ms=window_ms,
                status="low_energy_audio",
            )

        raw_probs = await asyncio.to_thread(
            _run_inference_sync,
            audio_array,
            sample_rate,
        )

        smoothed_probs = _apply_ema(state, raw_probs, alpha=EMA_ALPHA)
        label_str = _argmax_label_from_probs(smoothed_probs)
        confidence = float(np.max(smoothed_probs))

        emotion_label = EmotionLabel(label_str)

        result = EmotionResult(
            emotion=emotion_label.value,
            confidence=confidence,
        )

        logger.debug(
            "Emotion detected: %s (confidence=%.3f, window_ms=%d, rms=%.5f, gate=%.5f)",
            result.emotion,
            result.confidence,
            window_ms,
            rms,
            dynamic_gate,
        )

        if session_id and _should_log_this_window(state):
            await _enqueue_emotion_log(
                {
                    "user_id": user_id,
                    "session_id": session_id,
                    "emotion": result.emotion,
                    "confidence": result.confidence,
                    "detected_at": timestamp,
                    # window_ms dropped — not a column in the table
                }
            )

        return {
            "label": result.emotion,
            "confidence": result.confidence,
            "timestamp": timestamp,
            "window_ms": window_ms,
        }

    except Exception as exc:
        logger.exception("Emotion detection error")
        return {
            "label": FALLBACK_LABEL,
            "confidence": 0.0,
            "timestamp": timestamp,
            "window_ms": 0,
            "error": str(exc),
        }


# ADK FunctionTool wrapper
detect_emotion_tool = FunctionTool(func=detect_emotion)
