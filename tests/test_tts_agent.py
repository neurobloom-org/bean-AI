"""Targeted tests for agents/tts/agent.py — validate _validate_tts_chunk fix."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from shared.schemas import TTSChunk
from agents.tts.agent import _validate_tts_chunk


# ── _validate_tts_chunk tests ────────────────────────────────────────────────

def test_validate_tts_chunk_passes_valid_chunk():
    """Valid TTSChunk with audio_chunk bytes should pass through unchanged."""
    chunk = TTSChunk(audio_chunk=b"\x00\x01\x02", is_final=False, turn_id="abc")
    result = _validate_tts_chunk(chunk)
    assert result is chunk


def test_validate_tts_chunk_passes_final_empty_chunk():
    """Final chunk with empty audio_chunk is allowed (signals stream end)."""
    chunk = TTSChunk(audio_chunk=b"", is_final=True, turn_id="abc")
    result = _validate_tts_chunk(chunk)
    assert result is chunk


def test_validate_tts_chunk_rejects_non_final_empty_audio():
    """Non-final chunk with empty audio_chunk bytes should raise ValueError."""
    chunk = TTSChunk(audio_chunk=b"", is_final=False, turn_id="abc")
    with pytest.raises(ValueError, match="cannot be empty"):
        _validate_tts_chunk(chunk)


def test_validate_tts_chunk_rejects_wrong_type():
    """Passing a plain dict instead of TTSChunk should raise TypeError."""
    fake_chunk = {"audio_chunk": b"\x00", "is_final": False, "turn_id": "abc"}
    with pytest.raises(TypeError, match="TTSChunk instances"):
        _validate_tts_chunk(fake_chunk)


def test_validate_tts_chunk_rejects_none():
    """Passing None should raise TypeError."""
    with pytest.raises(TypeError, match="TTSChunk instances"):
        _validate_tts_chunk(None)