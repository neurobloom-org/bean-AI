"""Unit tests for casual_chat_agent sanitize pipeline."""

from agents.casual_chat.agent import (
    sanitize_response,
    fallback_response,
)


# ── sanitize_response ────────────────────────────────────────────────────────

def test_strips_emojis():
    result = sanitize_response("Hey that's cool! 😊", "neutral")
    assert "😊" not in result

def test_strips_asterisk_actions():
    result = sanitize_response("*leans forward* That sounds tough.", "sad")
    assert "*" not in result

def test_limits_to_two_sentences():
    long = "First sentence. Second sentence. Third sentence."
    result = sanitize_response(long, "neutral")
    assert result.count(".") <= 2

def test_scrubs_as_an_ai():
    result = sanitize_response("As an AI, I understand how you feel.", "neutral")
    assert "as an ai" not in result.lower()

def test_question_gets_question_mark():
    result = sanitize_response("What happened today", "neutral")
    assert result.strip().endswith("?")

def test_happy_emotion_gets_exclamation():
    result = sanitize_response("That is so great", "happy")
    assert "!" in result

def test_sad_emotion_gets_period():
    result = sanitize_response("I hear you", "sad")
    assert result.strip().endswith(".")

def test_empty_input_returns_fallback():
    result = sanitize_response("", "neutral")
    assert len(result) > 0

def test_non_string_input_returns_fallback():
    result = sanitize_response(None, "neutral")
    assert len(result) > 0

def test_unknown_emotion_normalizes_to_neutral():
    # Should not crash, should behave like neutral
    result = sanitize_response("Hey there", "confused")
    assert len(result) > 0


# ── fallback_response ────────────────────────────────────────────────────────

def test_fallback_sad():
    result = fallback_response("sad")
    assert "here" in result.lower() or "wrong" in result.lower()

def test_fallback_happy():
    result = fallback_response("happy")
    assert "!" in result

def test_fallback_angry():
    result = fallback_response("angry")
    assert "?" in result or "hear" in result.lower()

def test_fallback_unknown_emotion():
    result = fallback_response("confused")
    assert len(result) > 0