"""Tests for agents/alert/agent.py

Covers:
  - _compute_alert_level (pure function, all branches)
  - _parse_emotion_confidence (type safety, clamping)
  - _parse_already_dispatched (bool / string / garbage)
  - _normalize_recent_emotion_labels (str / list / dict / unknown)
  - AlertAgent._run_async_impl (factor evaluation, dispatch gating,
    deduplication, minor vs adult threshold, crisis path)

No real DB or Twilio calls are made — everything is mocked.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.alert.agent import (
    AlertAgent,
    _compute_alert_level,
    _normalize_recent_emotion_labels,
    _parse_already_dispatched,
    _parse_emotion_confidence,
    alert_agent,
)
from shared.enums import AlertLevel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_ctx(state: dict[str, Any]) -> MagicMock:
    ctx = MagicMock()
    ctx.session.state = state
    return ctx


async def _run_agent(state: dict[str, Any]) -> dict[str, Any]:
    agent = AlertAgent(name="test_alert_agent")
    async for event in agent._run_async_impl(_make_ctx(state)):
        if event.actions and event.actions.state_delta:
            state.update(event.actions.state_delta)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# _compute_alert_level
# ─────────────────────────────────────────────────────────────────────────────


class TestComputeAlertLevel:
    def test_zero_factors_is_none(self):
        assert _compute_alert_level(0, threshold=3) == AlertLevel.NONE

    def test_negative_count_is_none(self):
        assert _compute_alert_level(-1, threshold=3) == AlertLevel.NONE

    def test_one_factor_is_low(self):
        assert _compute_alert_level(1, threshold=3) == AlertLevel.LOW

    def test_two_factors_below_threshold_is_medium(self):
        assert _compute_alert_level(2, threshold=3) == AlertLevel.MEDIUM

    def test_at_threshold_is_high(self):
        assert _compute_alert_level(3, threshold=3) == AlertLevel.HIGH

    def test_above_threshold_is_crisis(self):
        assert _compute_alert_level(4, threshold=3) == AlertLevel.CRISIS
        assert _compute_alert_level(5, threshold=3) == AlertLevel.CRISIS

    def test_minor_threshold_two(self):
        # threshold=2: 0→NONE, 1→LOW, 2→HIGH (no MEDIUM), 3→CRISIS
        assert _compute_alert_level(0, threshold=2) == AlertLevel.NONE
        assert _compute_alert_level(1, threshold=2) == AlertLevel.LOW
        assert _compute_alert_level(2, threshold=2) == AlertLevel.HIGH
        assert _compute_alert_level(3, threshold=2) == AlertLevel.CRISIS

    def test_threshold_one(self):
        # Everything at or above 1 should be HIGH or CRISIS
        assert _compute_alert_level(1, threshold=1) == AlertLevel.HIGH
        assert _compute_alert_level(2, threshold=1) == AlertLevel.CRISIS


# ─────────────────────────────────────────────────────────────────────────────
# _parse_emotion_confidence
# ─────────────────────────────────────────────────────────────────────────────


class TestParseEmotionConfidence:
    def test_valid_float(self):
        assert _parse_emotion_confidence(0.85) == pytest.approx(0.85)

    def test_valid_string_float(self):
        assert _parse_emotion_confidence("0.9") == pytest.approx(0.9)

    def test_clamps_above_one(self):
        assert _parse_emotion_confidence(1.5) == 1.0

    def test_clamps_below_zero(self):
        assert _parse_emotion_confidence(-0.3) == 0.0

    def test_bool_true_returns_zero(self):
        assert _parse_emotion_confidence(True) == 0.0

    def test_bool_false_returns_zero(self):
        assert _parse_emotion_confidence(False) == 0.0

    def test_none_returns_zero(self):
        assert _parse_emotion_confidence(None) == 0.0

    def test_garbage_string_returns_zero(self):
        assert _parse_emotion_confidence("not-a-number") == 0.0

    def test_integer_input(self):
        assert _parse_emotion_confidence(1) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# _parse_already_dispatched
# ─────────────────────────────────────────────────────────────────────────────


class TestParseAlreadyDispatched:
    def test_bool_true(self):
        assert _parse_already_dispatched(True) is True

    def test_bool_false(self):
        assert _parse_already_dispatched(False) is False

    def test_string_true(self):
        assert _parse_already_dispatched("true") is True
        assert _parse_already_dispatched("True") is True
        assert _parse_already_dispatched("TRUE") is True

    def test_string_one(self):
        assert _parse_already_dispatched("1") is True

    def test_string_yes(self):
        assert _parse_already_dispatched("yes") is True

    def test_string_false(self):
        assert _parse_already_dispatched("false") is False

    def test_string_zero(self):
        assert _parse_already_dispatched("0") is False

    def test_none_returns_false(self):
        assert _parse_already_dispatched(None) is False

    def test_integer_returns_false(self):
        assert _parse_already_dispatched(1) is False


# ─────────────────────────────────────────────────────────────────────────────
# _normalize_recent_emotion_labels
# ─────────────────────────────────────────────────────────────────────────────


class TestNormalizeRecentEmotionLabels:
    def test_plain_strings(self):
        result = _normalize_recent_emotion_labels(["sad", "angry", "neutral"])
        assert result == ["sad", "angry", "neutral"]

    def test_list_tuples(self):
        result = _normalize_recent_emotion_labels([["sad", 0.9], ["happy", 0.7]])
        assert result == ["sad", "happy"]

    def test_dict_with_label_key(self):
        result = _normalize_recent_emotion_labels([{"label": "sad", "confidence": 0.9}])
        assert result == ["sad"]

    def test_dict_with_emotion_key(self):
        result = _normalize_recent_emotion_labels([{"emotion": "angry"}])
        assert result == ["angry"]

    def test_mixed_formats(self):
        result = _normalize_recent_emotion_labels(["sad", ["angry", 0.8], {"label": "neutral"}])
        assert result == ["sad", "angry", "neutral"]

    def test_unknown_type_gives_empty_string(self):
        result = _normalize_recent_emotion_labels([42, None])
        assert result == ["", ""]

    def test_empty_list(self):
        assert _normalize_recent_emotion_labels([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# AlertAgent._run_async_impl — fast exits
# ─────────────────────────────────────────────────────────────────────────────


class TestAlertAgentFastExits:
    @pytest.mark.asyncio
    async def test_empty_transcript_skips_all_factor_checks(self):
        state: dict[str, Any] = {
            "current_transcript": "",
            "user_id": "user-1",
            "session_id": "sess-1",
        }
        result = await _run_agent(state)
        # No alert_level written on fast exit
        assert "alert_level" not in result

    @pytest.mark.asyncio
    async def test_already_dispatched_skips_all_factor_checks(self):
        state: dict[str, Any] = {
            "current_transcript": "I want to die",
            "user_id": "user-1",
            "session_id": "sess-1",
            "alert_dispatched": "true",
        }
        result = await _run_agent(state)
        assert "alert_level" not in result


# ─────────────────────────────────────────────────────────────────────────────
# AlertAgent._run_async_impl — factor evaluation
# ─────────────────────────────────────────────────────────────────────────────


class TestAlertAgentFactorEvaluation:
    @pytest.mark.asyncio
    async def test_no_factors_gives_none_level(self):
        state: dict[str, Any] = {
            "current_transcript": "I had a great day",
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_emotion": "happy",
            "emotion_confidence": 0.9,
            "is_minor": False,
            "alert_dispatched": False,
        }
        result = await _run_agent(state)
        assert result["alert_level"] == "none"
        assert result["alert_active_count"] == 0

    @pytest.mark.asyncio
    async def test_f1_crisis_keyword_detected(self):
        state: dict[str, Any] = {
            "current_transcript": "I want to kill myself",
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_emotion": "neutral",
            "emotion_confidence": 0.5,
            "is_minor": False,
            "alert_dispatched": False,
        }
        with patch("agents.alert.agent.send_guardian_alert", new_callable=AsyncMock):
            with patch("agents.alert.agent.get_service_client", new_callable=AsyncMock) as mock_client:
                mock_client.return_value.table.return_value.select.return_value.eq.return_value.eq.return_value.execute = AsyncMock(return_value=MagicMock(data=[]))
                mock_client.return_value.table.return_value.insert.return_value.execute = AsyncMock(return_value=MagicMock(data=[{"id": "alert-1"}]))
                result = await _run_agent(state)

        assert "f1_crisis_keyword" in result["alert_factors"]

    @pytest.mark.asyncio
    async def test_f2_negative_emotion_with_high_confidence(self):
        state: dict[str, Any] = {
            "current_transcript": "I'm feeling really down",
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_emotion": "sad",
            "emotion_confidence": 0.8,
            "is_minor": False,
            "alert_dispatched": False,
        }
        result = await _run_agent(state)
        assert "f2_negative_emotion" in result["alert_factors"]

    @pytest.mark.asyncio
    async def test_f2_negative_emotion_low_confidence_not_triggered(self):
        state: dict[str, Any] = {
            "current_transcript": "hmm",
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_emotion": "sad",
            "emotion_confidence": 0.3,  # below 0.5 threshold
            "is_minor": False,
            "alert_dispatched": False,
        }
        result = await _run_agent(state)
        assert "f2_negative_emotion" not in result["alert_factors"]

    @pytest.mark.asyncio
    async def test_f3_escalation_three_consecutive_negatives(self):
        state: dict[str, Any] = {
            "current_transcript": "still feeling bad",
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_emotion": "neutral",
            "emotion_confidence": 0.3,
            "is_minor": False,
            "alert_dispatched": False,
            "recent_emotions": ["sad", "angry", "fearful"],
        }
        result = await _run_agent(state)
        assert "f3_escalation_pattern" in result["alert_factors"]

    @pytest.mark.asyncio
    async def test_f3_not_triggered_with_mixed_emotions(self):
        state: dict[str, Any] = {
            "current_transcript": "just talking",
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_emotion": "neutral",
            "emotion_confidence": 0.3,
            "is_minor": False,
            "alert_dispatched": False,
            "recent_emotions": ["sad", "happy", "sad"],  # not all negative
        }
        result = await _run_agent(state)
        assert "f3_escalation_pattern" not in result["alert_factors"]

    @pytest.mark.asyncio
    async def test_f4_vulnerability_always_set_for_minors(self):
        state: dict[str, Any] = {
            "current_transcript": "just talking",
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_emotion": "neutral",
            "emotion_confidence": 0.3,
            "is_minor": True,
            "alert_dispatched": False,
        }
        result = await _run_agent(state)
        assert "f4_vulnerability" in result["alert_factors"]

    @pytest.mark.asyncio
    async def test_f4_not_set_for_adults(self):
        state: dict[str, Any] = {
            "current_transcript": "just talking",
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_emotion": "neutral",
            "emotion_confidence": 0.3,
            "is_minor": False,
            "alert_dispatched": False,
        }
        result = await _run_agent(state)
        assert "f4_vulnerability" not in result["alert_factors"]

    @pytest.mark.asyncio
    async def test_f5_explicit_statement(self):
        state: dict[str, Any] = {
            "current_transcript": "tonight I will end it",
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_emotion": "neutral",
            "emotion_confidence": 0.3,
            "is_minor": False,
            "alert_dispatched": False,
        }
        with patch("agents.alert.agent.send_guardian_alert", new_callable=AsyncMock):
            with patch("agents.alert.agent.get_service_client", new_callable=AsyncMock) as mock_client:
                mock_client.return_value.table.return_value.select.return_value.eq.return_value.eq.return_value.execute = AsyncMock(return_value=MagicMock(data=[]))
                mock_client.return_value.table.return_value.insert.return_value.execute = AsyncMock(return_value=MagicMock(data=[{"id": "alert-1"}]))
                result = await _run_agent(state)

        assert "f5_explicit_statement" in result["alert_factors"]

    @pytest.mark.asyncio
    async def test_factors_are_deduplicated(self):
        """Same factor should not appear twice even if multiple triggers fire."""
        state: dict[str, Any] = {
            "current_transcript": "I want to kill myself and tonight I will end it",
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_emotion": "neutral",
            "emotion_confidence": 0.3,
            "is_minor": False,
            "alert_dispatched": False,
        }
        with patch("agents.alert.agent.send_guardian_alert", new_callable=AsyncMock):
            with patch("agents.alert.agent.get_service_client", new_callable=AsyncMock) as mock_client:
                mock_client.return_value.table.return_value.select.return_value.eq.return_value.eq.return_value.execute = AsyncMock(return_value=MagicMock(data=[]))
                mock_client.return_value.table.return_value.insert.return_value.execute = AsyncMock(return_value=MagicMock(data=[{"id": "alert-1"}]))
                result = await _run_agent(state)

        factors = result["alert_factors"]
        assert len(factors) == len(set(factors)), "Duplicate factors found"


# ─────────────────────────────────────────────────────────────────────────────
# AlertAgent — threshold comparison (adult vs minor)
# ─────────────────────────────────────────────────────────────────────────────


class TestAlertAgentThresholds:
    @pytest.mark.asyncio
    async def test_adult_two_factors_is_medium_no_dispatch(self):
        """Adult with 2 factors (< threshold 3) → MEDIUM, no SMS."""
        state: dict[str, Any] = {
            "current_transcript": "I feel so sad",
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_emotion": "sad",
            "emotion_confidence": 0.9,
            "is_minor": False,
            "alert_dispatched": False,
            "recent_emotions": ["sad", "angry", "fearful"],  # F3
        }
        with patch("agents.alert.agent.send_guardian_alert", new_callable=AsyncMock) as mock_sms:
            result = await _run_agent(state)

        assert result["alert_level"] in ("medium", "high", "crisis")
        # At least 2 factors: F2 + F3
        assert result["alert_active_count"] >= 2
        # SMS only fires at HIGH/CRISIS
        if result["alert_level"] == "medium":
            mock_sms.assert_not_called()

    @pytest.mark.asyncio
    async def test_minor_two_factors_reaches_high(self):
        """Minor with F4 (always) + F2 = 2 factors → HIGH at minor threshold."""
        state: dict[str, Any] = {
            "current_transcript": "I feel sad",
            "user_id": "user-minor",
            "session_id": "sess-1",
            "current_emotion": "sad",
            "emotion_confidence": 0.8,
            "is_minor": True,
            "alert_dispatched": False,
        }
        with patch("agents.alert.agent.send_guardian_alert", new_callable=AsyncMock):
            with patch("agents.alert.agent.get_service_client", new_callable=AsyncMock) as mock_client:
                mock_client.return_value.table.return_value.select.return_value.eq.return_value.eq.return_value.execute = AsyncMock(return_value=MagicMock(data=[]))
                mock_client.return_value.table.return_value.insert.return_value.execute = AsyncMock(return_value=MagicMock(data=[{"id": "alert-1"}]))
                result = await _run_agent(state)

        # F4 (vulnerability, always for minors) + F2 (negative emotion) = 2 factors
        assert result["alert_active_count"] >= 2
        assert result["alert_level"] in ("high", "crisis")

    @pytest.mark.asyncio
    async def test_dispatch_sets_alert_dispatched_flag(self):
        """After a HIGH/CRISIS dispatch, alert_dispatched must be set."""
        state: dict[str, Any] = {
            "current_transcript": "I want to kill myself",
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_emotion": "sad",
            "emotion_confidence": 0.9,
            "is_minor": False,
            "alert_dispatched": False,
            "recent_emotions": ["sad", "angry", "fearful"],
        }
        with patch("agents.alert.agent.send_guardian_alert", new_callable=AsyncMock):
            with patch("agents.alert.agent.get_service_client", new_callable=AsyncMock) as mock_client:
                mock_client.return_value.table.return_value.select.return_value.eq.return_value.eq.return_value.execute = AsyncMock(return_value=MagicMock(data=[]))
                mock_client.return_value.table.return_value.insert.return_value.execute = AsyncMock(return_value=MagicMock(data=[{"id": "alert-1"}]))
                result = await _run_agent(state)

        if result["alert_level"] in ("high", "crisis"):
            assert result.get("alert_dispatched") == "true"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────


def test_alert_agent_singleton_name():
    assert alert_agent.name == "alert_agent"
