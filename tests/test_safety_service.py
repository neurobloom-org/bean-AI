"""Unit tests for services/safety_service.py"""

import sys
from unittest.mock import MagicMock

# Mock external dependencies
sys.modules["google"] = MagicMock()
sys.modules["google.genai"] = MagicMock()
sys.modules["google.genai.types"] = MagicMock()
sys.modules["google.adk"] = MagicMock()
sys.modules["google.adk.tools"] = MagicMock()
sys.modules["twilio"] = MagicMock()
sys.modules["twilio.rest"] = MagicMock()
sys.modules["supabase"] = MagicMock()
sys.modules["pydantic_settings"] = MagicMock()

from services.safety_service import (  # noqa: E402
    check_crisis_keywords,
    check_explicit_statement,
    get_post_alert_message,
)


def test_crisis_keywords_detected():
    has_crisis, matched = check_crisis_keywords("I want to kill myself")
    assert has_crisis is True
    assert len(matched) > 0


def test_crisis_keywords_not_detected():
    has_crisis, matched = check_crisis_keywords("I had a great day today")
    assert has_crisis is False
    assert len(matched) == 0


def test_crisis_keywords_case_insensitive():
    has_crisis, matched = check_crisis_keywords("I WANT TO DIE")
    assert has_crisis is True


def test_explicit_statement_detected():
    has_explicit, matched = check_explicit_statement("I have a plan tonight")
    assert has_explicit is True
    assert len(matched) > 0


def test_explicit_statement_not_detected():
    has_explicit, matched = check_explicit_statement("I feel a bit sad today")
    assert has_explicit is False
    assert len(matched) == 0


def test_get_post_alert_message_returns_string():
    msg = get_post_alert_message()
    assert isinstance(msg, str)
    assert len(msg) > 0


def test_get_post_alert_message_varies():
    messages = set()
    for _ in range(20):
        messages.add(get_post_alert_message())
    assert len(messages) > 1


def test_empty_text_no_crisis():
    has_crisis, matched = check_crisis_keywords("")
    assert has_crisis is False
    assert matched == []


def test_minor_flag_does_not_break():
    has_crisis, matched = check_crisis_keywords("I want to die", is_minor=True)
    assert has_crisis is True
