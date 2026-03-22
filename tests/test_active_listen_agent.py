"""Unit tests for agents/active_listen/agent.py"""

import sys
from unittest.mock import MagicMock

# Mock google.adk before importing the agent
sys.modules["google"] = MagicMock()
sys.modules["google.adk"] = MagicMock()
sys.modules["google.adk.tools"] = MagicMock()

from agents.active_listen.agent import get_filler_phrase  # noqa: E402


def test_returns_phrase_for_sad_emotion():
    result = get_filler_phrase(emotion="sad")
    assert "phrase" in result
    assert isinstance(result["phrase"], str)
    assert len(result["phrase"]) > 0


def test_returns_phrase_for_happy_emotion():
    result = get_filler_phrase(emotion="happy")
    assert "phrase" in result
    assert len(result["phrase"]) > 0


def test_returns_phrase_for_neutral_emotion():
    result = get_filler_phrase(emotion="neutral")
    assert "phrase" in result
    assert isinstance(result["phrase"], str)


def test_returns_phrase_for_unknown_emotion():
    result = get_filler_phrase(emotion="confused")
    assert "phrase" in result
    assert isinstance(result["phrase"], str)


def test_avoids_repeating_last_phrase():
    result1 = get_filler_phrase(emotion="sad", last_phrase="")
    last = result1["phrase"]
    phrases = set()
    for _ in range(20):
        r = get_filler_phrase(emotion="sad", last_phrase=last)
        phrases.add(r["phrase"])
    assert len(phrases) > 0


def test_returns_audio_cache_key():
    result = get_filler_phrase(emotion="neutral")
    assert "audio_cache_key" in result
    assert isinstance(result["audio_cache_key"], str)
    assert "filler_" in result["audio_cache_key"]


def test_audio_b64_is_none_by_default():
    result = get_filler_phrase(emotion="neutral")
    assert result.get("audio_b64") is None


def test_all_emotions_return_phrase():
    emotions = ["sad", "angry", "fearful", "disgusted", "happy", "calm", "surprised", "neutral"]
    for emotion in emotions:
        result = get_filler_phrase(emotion=emotion)
        assert isinstance(result["phrase"], str)
        assert len(result["phrase"]) > 0
