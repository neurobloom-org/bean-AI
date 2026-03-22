"""Tests for services/safety_service.py

Covers:
  - check_crisis_keywords (matching, case insensitivity, partial phrases)
  - check_explicit_statement (explicit intent detection)
  - get_post_alert_message (non-empty, random from list)
  - SafetyService.assess_turn (mocked LLM + DB)

No real LLM, DB, or Twilio calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.safety_service import (
    CRISIS_KEYWORDS,
    EXPLICIT_STATEMENT_KEYWORDS,
    SafetyService,
    check_crisis_keywords,
    check_explicit_statement,
    get_post_alert_message,
)
from shared.exceptions import CrisisDetectedError


# ─────────────────────────────────────────────────────────────────────────────
# check_crisis_keywords
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckCrisisKeywords:
    def test_detects_kill_myself(self):
        detected, matches = check_crisis_keywords("I want to kill myself")
        assert detected is True
        assert any("kill myself" in m for m in matches)

    def test_detects_suicide(self):
        detected, _ = check_crisis_keywords("I'm thinking about suicide")
        assert detected is True

    def test_detects_want_to_die(self):
        detected, _ = check_crisis_keywords("I want to die so badly")
        assert detected is True

    def test_case_insensitive(self):
        detected, _ = check_crisis_keywords("I WANT TO DIE")
        assert detected is True

    def test_no_keyword_returns_false(self):
        detected, matches = check_crisis_keywords("I had a great day at school")
        assert detected is False
        assert matches == []

    def test_empty_string_returns_false(self):
        detected, matches = check_crisis_keywords("")
        assert detected is False
        assert matches == []

    def test_partial_match_in_sentence(self):
        """Keywords embedded in a longer sentence should still fire."""
        detected, _ = check_crisis_keywords("She told me she wants to hurt herself")
        assert detected is True

    def test_returns_all_matched_keywords(self):
        detected, matches = check_crisis_keywords(
            "I want to die and I want to hurt myself"
        )
        assert detected is True
        assert len(matches) >= 2

    def test_is_minor_flag_does_not_break(self):
        """is_minor parameter should not cause errors."""
        detected, _ = check_crisis_keywords("normal text", is_minor=True)
        assert detected is False


# ─────────────────────────────────────────────────────────────────────────────
# check_explicit_statement
# ─────────────────────────────────────────────────────────────────────────────


class TestCheckExplicitStatement:
    def test_detects_tonight_i(self):
        detected, matches = check_explicit_statement("Tonight I will end everything")
        assert detected is True
        assert any("tonight i" in m for m in matches)

    def test_detects_i_have_a_plan(self):
        detected, _ = check_explicit_statement("I have a plan to hurt myself")
        assert detected is True

    def test_detects_i_already_cut(self):
        detected, _ = check_explicit_statement("I already cut my arm")
        assert detected is True

    def test_no_explicit_statement_returns_false(self):
        detected, matches = check_explicit_statement("I feel really sad today")
        assert detected is False
        assert matches == []

    def test_case_insensitive(self):
        detected, _ = check_explicit_statement("TONIGHT I WILL DO IT")
        assert detected is True

    def test_empty_text_returns_false(self):
        detected, matches = check_explicit_statement("")
        assert detected is False
        assert matches == []


# ─────────────────────────────────────────────────────────────────────────────
# get_post_alert_message
# ─────────────────────────────────────────────────────────────────────────────


class TestGetPostAlertMessage:
    def test_returns_non_empty_string(self):
        msg = get_post_alert_message()
        assert isinstance(msg, str)
        assert len(msg) > 0

    def test_returns_different_messages_over_calls(self):
        """With enough calls, we should see at least two different messages."""
        messages = {get_post_alert_message() for _ in range(30)}
        assert len(messages) > 1


# ─────────────────────────────────────────────────────────────────────────────
# SafetyService.assess_turn
# ─────────────────────────────────────────────────────────────────────────────


class TestSafetyServiceAssessTurn:
    @pytest.mark.asyncio
    async def test_no_risk_returns_none_level(self):
        service = SafetyService()

        mock_assessment = {
            "alert_level": "none",
            "factors": [],
            "requires_immediate_action": False,
            "suggested_response_type": "supportive",
        }

        with patch("services.safety_service.assess_safety", new_callable=AsyncMock, return_value=mock_assessment):
            with patch("services.safety_service.get_service_client", new_callable=AsyncMock):
                result = await service.assess_turn(
                    user_id="user-1",
                    session_id="sess-1",
                    user_text="I had a nice day",
                    emotion="happy",
                    emotion_trend=["happy", "neutral"],
                    turn_number=1,
                )

        assert result["alert_level"] == "none"

    @pytest.mark.asyncio
    async def test_crisis_keyword_raises_crisis_detected_error(self):
        service = SafetyService()

        mock_assessment = {
            "alert_level": "crisis",
            "factors": ["f1_crisis_keyword"],
            "requires_immediate_action": True,
            "suggested_response_type": "crisis_resources",
        }

        with patch("services.safety_service.assess_safety", new_callable=AsyncMock, return_value=mock_assessment):
            with patch("services.safety_service.get_service_client", new_callable=AsyncMock) as mock_client:
                mock_client.return_value.table.return_value.insert.return_value.execute = AsyncMock(
                    return_value=MagicMock(data=[{"id": "alert-1"}])
                )
                with pytest.raises(CrisisDetectedError):
                    await service.assess_turn(
                        user_id="user-1",
                        session_id="sess-1",
                        user_text="I want to kill myself",
                        emotion="sad",
                        emotion_trend=["sad", "sad", "sad"],
                        turn_number=3,
                    )

    @pytest.mark.asyncio
    async def test_llm_timeout_falls_back_gracefully(self):
        service = SafetyService()

        import asyncio

        with patch("services.safety_service.assess_safety", new_callable=AsyncMock, side_effect=asyncio.TimeoutError()):
            with patch("services.safety_service.get_service_client", new_callable=AsyncMock):
                result = await service.assess_turn(
                    user_id="user-1",
                    session_id="sess-1",
                    user_text="I'm just bored",
                    emotion="neutral",
                    emotion_trend=["neutral"],
                    turn_number=1,
                )

        assert result["alert_level"] == "none"

    @pytest.mark.asyncio
    async def test_vulnerability_flag_adds_f4(self):
        service = SafetyService()

        mock_assessment = {
            "alert_level": "low",
            "factors": [],
        }

        with patch("services.safety_service.assess_safety", new_callable=AsyncMock, return_value=mock_assessment):
            with patch("services.safety_service.get_service_client", new_callable=AsyncMock) as mock_client:
                mock_client.return_value.table.return_value.insert.return_value.execute = AsyncMock(
                    return_value=MagicMock(data=[{"id": "alert-1"}])
                )
                result = await service.assess_turn(
                    user_id="user-1",
                    session_id="sess-1",
                    user_text="feeling a bit off",
                    emotion="neutral",
                    emotion_trend=["neutral"],
                    turn_number=1,
                    vulnerability_flag=True,
                )

        assert "f4_vulnerability" in result["factors"]

    @pytest.mark.asyncio
    async def test_escalation_pattern_adds_f3(self):
        service = SafetyService()

        mock_assessment = {"alert_level": "none", "factors": []}

        with patch("services.safety_service.assess_safety", new_callable=AsyncMock, return_value=mock_assessment):
            with patch("services.safety_service.get_service_client", new_callable=AsyncMock):
                result = await service.assess_turn(
                    user_id="user-1",
                    session_id="sess-1",
                    user_text="still feeling down",
                    emotion="neutral",
                    emotion_trend=["sad", "angry", "fearful"],  # 3 consecutive negatives
                    turn_number=4,
                )

        assert "f3_escalation_pattern" in result["factors"]


# ─────────────────────────────────────────────────────────────────────────────
# Keyword set sanity checks
# ─────────────────────────────────────────────────────────────────────────────


def test_crisis_keywords_all_lowercase():
    for kw in CRISIS_KEYWORDS:
        assert kw == kw.lower(), f"Crisis keyword not lowercase: {kw!r}"


def test_explicit_statement_keywords_all_lowercase():
    for kw in EXPLICIT_STATEMENT_KEYWORDS:
        assert kw == kw.lower(), f"Explicit keyword not lowercase: {kw!r}"


def test_keyword_sets_are_non_empty():
    assert len(CRISIS_KEYWORDS) > 0
    assert len(EXPLICIT_STATEMENT_KEYWORDS) > 0
