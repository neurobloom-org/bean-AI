"""Unit tests for agents/stt/agent.py.

Deepgram WebSocket is fully mocked — no real API calls, no internet required.
Run with: make test
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agents.stt.agent import (
    create_deepgram_connection,
    close_deepgram_connection,
    stream_audio_chunk,
    deepgram_transcribe,
    _active_sessions,
    _pending_session_tasks,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_mock_connection(is_connected: bool = True) -> MagicMock:
    """Build a fake DeepgramConnection with controllable state."""
    conn = MagicMock()
    conn.is_connected = is_connected
    conn.connect = AsyncMock()
    conn.close = AsyncMock()
    conn.send_audio = AsyncMock()
    conn.send_audio_b64 = AsyncMock()
    conn.send_keepalive = AsyncMock()
    return conn


@pytest.fixture(autouse=True)
async def clean_registry():
    """Wipe the module-level session registry before and after every test.
    
    Without this, state leaks between tests and causes false passes/failures.
    """
    _active_sessions.clear()
    _pending_session_tasks.clear()
    yield
    _active_sessions.clear()
    _pending_session_tasks.clear()


# ─────────────────────────────────────────────────────────────────────────────
# stream_audio_chunk — the hot path
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stream_audio_no_session_returns_no_connection():
    result = await stream_audio_chunk("session-abc", b"\x00\x01")
    assert result["status"] == "no_connection"


@pytest.mark.asyncio
async def test_stream_audio_empty_session_id_returns_invalid_input():
    result = await stream_audio_chunk("", b"\x00\x01")
    assert result["status"] == "invalid_input"


@pytest.mark.asyncio
async def test_stream_audio_empty_chunk_returns_invalid_input():
    result = await stream_audio_chunk("session-abc", b"")
    assert result["status"] == "invalid_input"


@pytest.mark.asyncio
async def test_stream_audio_disconnected_connection_returns_disconnected():
    from agents.stt.agent import DeepgramSession
    conn = make_mock_connection(is_connected=False)
    session = DeepgramSession(session_id="session-abc", connection=conn)
    _active_sessions["session-abc"] = session

    result = await stream_audio_chunk("session-abc", b"\x00\x01")
    assert result["status"] == "disconnected"


@pytest.mark.asyncio
async def test_stream_audio_sends_and_returns_audio_sent():
    from agents.stt.agent import DeepgramSession
    conn = make_mock_connection(is_connected=True)
    session = DeepgramSession(session_id="session-abc", connection=conn)
    _active_sessions["session-abc"] = session

    result = await stream_audio_chunk("session-abc", b"\x00\x01\x02")

    assert result["status"] == "audio_sent"
    assert result["bytes"] == 3
    conn.send_audio.assert_awaited_once_with(b"\x00\x01\x02")


# ─────────────────────────────────────────────────────────────────────────────
# deepgram_transcribe — the ADK fallback path
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_deepgram_transcribe_empty_session_id():
    result = await deepgram_transcribe("", "dGVzdA==")
    assert result["status"] == "invalid_input"


@pytest.mark.asyncio
async def test_deepgram_transcribe_empty_audio_b64():
    result = await deepgram_transcribe("session-abc", "")
    assert result["status"] == "invalid_input"


@pytest.mark.asyncio
async def test_deepgram_transcribe_invalid_base64():
    result = await deepgram_transcribe("session-abc", "!!!not-valid-base64!!!")
    assert result["status"] == "invalid_input"
    assert "base64" in result["message"]


@pytest.mark.asyncio
async def test_deepgram_transcribe_no_connection():
    # valid base64 of b"\x00\x01\x02"
    result = await deepgram_transcribe("session-abc", "AAEC")
    assert result["status"] == "no_connection"


@pytest.mark.asyncio
async def test_deepgram_transcribe_sends_decoded_bytes():
    from agents.stt.agent import DeepgramSession
    conn = make_mock_connection(is_connected=True)
    session = DeepgramSession(session_id="session-abc", connection=conn)
    _active_sessions["session-abc"] = session

    # base64 of b"\x00\x01\x02" is "AAEC"
    result = await deepgram_transcribe("session-abc", "AAEC")

    assert result["status"] == "audio_sent"
    conn.send_audio.assert_awaited_once_with(b"\x00\x01\x02")


# ─────────────────────────────────────────────────────────────────────────────
# close_deepgram_connection
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_close_nonexistent_session_does_not_raise():
    # Must not raise — safe to call even if session never existed
    await close_deepgram_connection("session-does-not-exist")


@pytest.mark.asyncio
async def test_close_empty_session_id_does_not_raise():
    await close_deepgram_connection("")


@pytest.mark.asyncio
async def test_close_removes_session_from_registry():
    from agents.stt.agent import DeepgramSession
    conn = make_mock_connection(is_connected=True)
    session = DeepgramSession(session_id="session-abc", connection=conn)
    _active_sessions["session-abc"] = session

    await close_deepgram_connection("session-abc")

    assert "session-abc" not in _active_sessions
    conn.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_after_close_transcribe_returns_no_connection():
    from agents.stt.agent import DeepgramSession
    conn = make_mock_connection(is_connected=True)
    session = DeepgramSession(session_id="session-abc", connection=conn)
    _active_sessions["session-abc"] = session

    await close_deepgram_connection("session-abc")
    result = await deepgram_transcribe("session-abc", "AAEC")

    assert result["status"] == "no_connection"


# ─────────────────────────────────────────────────────────────────────────────
# create_deepgram_connection
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_raises_on_empty_session_id():
    with pytest.raises(ValueError, match="session_id must not be empty"):
        await create_deepgram_connection("", AsyncMock())


@pytest.mark.asyncio
async def test_create_stores_session_in_registry():
    mock_conn = make_mock_connection(is_connected=True)

    with patch(
        "agents.stt.agent.DeepgramConnection", return_value=mock_conn
    ):
        conn = await create_deepgram_connection(
            "session-abc", AsyncMock(), AsyncMock()
        )

    assert "session-abc" in _active_sessions
    assert conn is mock_conn


@pytest.mark.asyncio
async def test_create_reuses_existing_healthy_connection():
    from agents.stt.agent import DeepgramSession
    conn = make_mock_connection(is_connected=True)
    session = DeepgramSession(session_id="session-abc", connection=conn)
    _active_sessions["session-abc"] = session

    with patch("agents.stt.agent.DeepgramConnection") as mock_cls:
        returned = await create_deepgram_connection("session-abc", AsyncMock())

    # DeepgramConnection constructor should NOT have been called again
    mock_cls.assert_not_called()
    assert returned is conn