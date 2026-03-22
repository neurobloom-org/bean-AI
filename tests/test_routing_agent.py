"""Tests for agents/routing/agent.py

Covers every item in the branch guide Pre-PR checklist plus the
improvements added in the final version.

Run from the project root:
    pytest tests/test_routing_agent.py -v

No real LLM calls are made — generate_json is mocked throughout.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.routing.agent import (
    DEFAULT_CONFIDENCE,
    EMPTY_TRANSCRIPT_CONFIDENCE,
    MAX_TRANSCRIPT_CHARS,
    RoutingAgent,
    _apply_alert_confidence_floor,
    _normalize_emotion,
    _normalize_route_distribution,
    _RoutingDecision,
    _sanitize_user_text,
    routing_agent,
)
from shared.enums import RouteType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(state: dict[str, Any]) -> MagicMock:
    """Build a minimal fake InvocationContext with the given session state."""
    ctx = MagicMock()
    ctx.session.state = state
    return ctx


async def _run_agent(state: dict[str, Any]) -> dict[str, Any]:
    ctx = _make_ctx(state)
    agent = RoutingAgent(name="test_routing_agent")
    async for event in agent._run_async_impl(ctx):
        if event.actions and event.actions.state_delta:
            state.update(event.actions.state_delta)
    return state


def _make_llm_response(route: str, confidence: float) -> dict[str, Any]:
    return {"route": route, "confidence": confidence}


# ---------------------------------------------------------------------------
# Branch guide Pre-PR checklist
# ---------------------------------------------------------------------------


class TestBranchGuideChecklist:
    """These four tests map 1-to-1 to the checklist in the branch guide."""

    @pytest.mark.asyncio
    async def test_sad_transcript_routes_to_therapy(self) -> None:
        """Mock state with 'I feel really sad today' → route is 'therapy'."""
        state: dict[str, Any] = {
            "current_transcript": "I feel really sad today",
            "current_emotion": "sad",
            "turn_count": 1,
            "route_distribution": {},
        }
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            return_value=_make_llm_response("therapy", 0.95),
        ):
            result = await _run_agent(state)

        assert result["route"] == "therapy"

    @pytest.mark.asyncio
    async def test_music_transcript_routes_to_music(self) -> None:
        """Mock state with 'play some music' → route is 'music'."""
        state: dict[str, Any] = {
            "current_transcript": "play some music",
            "current_emotion": "neutral",
            "turn_count": 1,
            "route_distribution": {},
        }
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            return_value=_make_llm_response("music", 0.98),
        ):
            result = await _run_agent(state)

        assert result["route"] == "music"

    @pytest.mark.asyncio
    async def test_empty_transcript_defaults_to_casual_without_llm(self) -> None:
        """Empty transcript → defaults to 'casual' without calling LLM."""
        state: dict[str, Any] = {"current_transcript": ""}

        with patch(
            "agents.routing.agent.generate_json", new_callable=AsyncMock
        ) as mock_llm:
            result = await _run_agent(state)
            mock_llm.assert_not_called()

        assert result["route"] == RouteType.CASUAL.value
        assert result["routing_confidence"] == EMPTY_TRANSCRIPT_CONFIDENCE

    @pytest.mark.asyncio
    async def test_malformed_llm_response_falls_back_to_casual(self) -> None:
        """Malformed LLM response → falls back to 'casual', does not raise."""
        state: dict[str, Any] = {
            "current_transcript": "hello there",
            "current_emotion": "neutral",
        }

        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            side_effect=ValueError("LLM returned non-JSON"),
        ):
            # Must not raise
            result = await _run_agent(state)

        assert result["route"] == RouteType.CASUAL.value
        assert result["routing_used_fallback"] is True


# ---------------------------------------------------------------------------
# Alert confidence floor (new behavior)
# ---------------------------------------------------------------------------


class TestAlertConfidenceFloor:
    @pytest.mark.asyncio
    async def test_high_confidence_alert_passes_through(self) -> None:
        """Alert with confidence >= floor should stay as alert."""
        state: dict[str, Any] = {
            "current_transcript": "I want to hurt myself",
            "current_emotion": "sad",
        }
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            return_value=_make_llm_response("alert", 0.95),
        ):
            result = await _run_agent(state)

        assert result["route"] == "alert"
        assert result["routing_alert_suspected"] is True

    @pytest.mark.asyncio
    async def test_low_confidence_alert_downgrades_to_therapy(self) -> None:
        """Alert with confidence < floor should be downgraded to therapy."""
        state: dict[str, Any] = {
            "current_transcript": "I don't know what to do anymore",
            "current_emotion": "sad",
        }
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            return_value=_make_llm_response("alert", 0.60),
        ):
            result = await _run_agent(state)

        assert result["route"] == "therapy"
        # The LLM did suspect alert before the floor was applied
        assert result["routing_alert_suspected"] is True

    def test_apply_alert_confidence_floor_unit(self) -> None:
        """Direct unit test of the floor helper function."""
        high = _RoutingDecision(route=RouteType.ALERT, confidence=0.95)
        low = _RoutingDecision(route=RouteType.ALERT, confidence=0.60)
        therapy = _RoutingDecision(route=RouteType.THERAPY, confidence=0.90)

        result_high, downgraded_high = _apply_alert_confidence_floor(high)
        result_low, downgraded_low = _apply_alert_confidence_floor(low)
        result_therapy, downgraded_therapy = _apply_alert_confidence_floor(therapy)

        assert result_high.route == RouteType.ALERT
        assert downgraded_high is False

        assert result_low.route == RouteType.THERAPY
        assert downgraded_low is True
        assert result_low.confidence == 0.60  # confidence preserved after downgrade

        assert result_therapy.route == RouteType.THERAPY
        assert downgraded_therapy is False


# ---------------------------------------------------------------------------
# Retry behavior
# ---------------------------------------------------------------------------


class TestRetryBehavior:
    @pytest.mark.asyncio
    async def test_retries_on_transient_failure_and_succeeds(self) -> None:
        """First call fails, second call succeeds → route is used, not fallback."""
        state: dict[str, Any] = {
            "current_transcript": "remind me to call mum at 5pm",
            "current_emotion": "neutral",
        }
        call_count = 0

        async def flaky_llm(**kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient network error")
            return _make_llm_response("task", 0.92)

        with patch("agents.routing.agent.generate_json", side_effect=flaky_llm):
            with patch("agents.routing.agent.asyncio.sleep", new_callable=AsyncMock):
                result = await _run_agent(state)

        assert result["route"] == "task"
        assert result["routing_used_fallback"] is False
        assert result["routing_attempt_count"] == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_falls_back(self) -> None:
        """All attempts fail → falls back to casual, records attempt count."""
        state: dict[str, Any] = {
            "current_transcript": "hello",
            "current_emotion": "neutral",
        }
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            side_effect=RuntimeError("persistent failure"),
        ):
            with patch("agents.routing.agent.asyncio.sleep", new_callable=AsyncMock):
                result = await _run_agent(state)

        assert result["route"] == RouteType.CASUAL.value
        assert result["routing_used_fallback"] is True
        assert result["routing_attempt_count"] == 2  # 1 + MAX_LLM_RETRIES


# ---------------------------------------------------------------------------
# Timeout behavior
# ---------------------------------------------------------------------------


class TestTimeout:
    @pytest.mark.asyncio
    async def test_timeout_triggers_fallback(self) -> None:
        """LLM call that times out → fallback to casual."""
        state: dict[str, Any] = {
            "current_transcript": "what time is it?",
            "current_emotion": "neutral",
        }

        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError(),
        ):
            with patch("agents.routing.agent.asyncio.sleep", new_callable=AsyncMock):
                result = await _run_agent(state)

        assert result["route"] == RouteType.CASUAL.value
        assert result["routing_used_fallback"] is True
        assert result["routing_failure_reason"] == "timeout"

    @pytest.mark.asyncio
    async def test_timeout_triggers_fallback_retry(self) -> None:
        """LLM call that times out → fallback to casual."""
        state: dict[str, Any] = {
            "current_transcript": "what time is it?",
            "current_emotion": "neutral",
        }

        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError(),
        ):
            with patch("agents.routing.agent.asyncio.sleep", new_callable=AsyncMock):
                result = await _run_agent(state)

        assert result["route"] == RouteType.CASUAL.value
        assert result["routing_used_fallback"] is True
        assert result["routing_failure_reason"] == "timeout"


# ---------------------------------------------------------------------------
# Observability — diagnostics always written
# ---------------------------------------------------------------------------


class TestDiagnosticsAlwaysWritten:
    @pytest.mark.asyncio
    async def test_diagnostics_written_on_success(self) -> None:
        state: dict[str, Any] = {
            "current_transcript": "I'm feeling anxious",
            "current_emotion": "fearful",
        }
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            return_value=_make_llm_response("therapy", 0.88),
        ):
            result = await _run_agent(state)

        assert result["routing_used_fallback"] is False
        assert result["routing_failure_reason"] == ""
        assert result["routing_attempt_count"] == 1
        assert isinstance(result["routing_latency_ms"], float)
        assert result["routing_latency_ms"] >= 0.0

    @pytest.mark.asyncio
    async def test_diagnostics_written_on_empty_transcript(self) -> None:
        state: dict[str, Any] = {"current_transcript": "   "}
        await _run_agent(state)

        assert "routing_used_fallback" in state
        assert "routing_failure_reason" in state
        assert state["routing_failure_reason"] == "empty_transcript"

    @pytest.mark.asyncio
    async def test_diagnostics_written_on_failure(self) -> None:
        state: dict[str, Any] = {"current_transcript": "hi"}
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            with patch("agents.routing.agent.asyncio.sleep", new_callable=AsyncMock):
                result = await _run_agent(state)

        assert result["routing_used_fallback"] is True
        assert result["routing_failure_reason"] != ""


# ---------------------------------------------------------------------------
# Input sanitization
# ---------------------------------------------------------------------------


class TestInputSanitization:
    def test_sanitize_user_text_collapses_whitespace(self) -> None:
        assert _sanitize_user_text("hello\nworld") == "hello world"
        assert _sanitize_user_text("too   many   spaces") == "too many spaces"
        assert _sanitize_user_text("  leading and trailing  ") == "leading and trailing"

    @pytest.mark.asyncio
    async def test_transcript_is_truncated_to_max_chars(self) -> None:
        """Transcripts longer than MAX_TRANSCRIPT_CHARS are truncated before LLM."""
        long_text = "a" * (MAX_TRANSCRIPT_CHARS + 100)
        state: dict[str, Any] = {"current_transcript": long_text}

        captured_prompt: list[str] = []

        async def capture_llm(task: str, prompt: str, system: str) -> dict[str, Any]:
            captured_prompt.append(prompt)
            return _make_llm_response("casual", 0.8)

        with patch("agents.routing.agent.generate_json", side_effect=capture_llm):
            await _run_agent(state)

        assert len(captured_prompt) == 1
        # The truncated text in the prompt must not exceed MAX_TRANSCRIPT_CHARS
        assert ("a" * (MAX_TRANSCRIPT_CHARS + 1)) not in captured_prompt[0]

    def test_normalize_emotion_unknown_string_preserved(self) -> None:
        """Unknown but non-empty emotion strings are kept, not reset to neutral."""
        assert _normalize_emotion("fearful_high") == "fearful_high"
        assert _normalize_emotion(None) == "neutral"
        assert _normalize_emotion("") == "neutral"
        assert _normalize_emotion(123) == "neutral"

    def test_normalize_route_distribution_filters_invalid_keys(self) -> None:
        raw = {"casual": 5, "therapy": 3, "unknown_route": 10, 42: 1}
        result = _normalize_route_distribution(raw)
        assert RouteType.CASUAL in result
        assert RouteType.THERAPY in result
        assert "unknown_route" not in str(result)
        assert 42 not in result

    def test_normalize_route_distribution_clamps_counts(self) -> None:
        from agents.routing.agent import MAX_ROUTE_DISTRIBUTION_COUNT

        raw = {"casual": MAX_ROUTE_DISTRIBUTION_COUNT + 9999, "therapy": -5}
        result = _normalize_route_distribution(raw)
        assert result[RouteType.CASUAL] == MAX_ROUTE_DISTRIBUTION_COUNT
        assert result[RouteType.THERAPY] == 0


# ---------------------------------------------------------------------------
# LLM response validation edge cases
# ---------------------------------------------------------------------------


class TestLLMResponseValidation:
    @pytest.mark.asyncio
    async def test_invalid_route_string_defaults_to_casual(self) -> None:
        state: dict[str, Any] = {"current_transcript": "hello"}
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            return_value={"route": "INVALID_ROUTE", "confidence": 0.9},
        ):
            result = await _run_agent(state)
        assert result["route"] == RouteType.CASUAL.value

    @pytest.mark.asyncio
    async def test_boolean_confidence_uses_default(self) -> None:
        """bool confidence (True/False) should not be accepted as 1.0/0.0."""
        state: dict[str, Any] = {"current_transcript": "hey"}
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            return_value={"route": "casual", "confidence": True},
        ):
            result = await _run_agent(state)
        assert result["routing_confidence"] == DEFAULT_CONFIDENCE

    @pytest.mark.asyncio
    async def test_out_of_range_confidence_clamped(self) -> None:
        state: dict[str, Any] = {"current_transcript": "hey"}
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            return_value={"route": "casual", "confidence": 99.9},
        ):
            result = await _run_agent(state)
        assert result["routing_confidence"] == 1.0

    @pytest.mark.asyncio
    async def test_non_dict_llm_response_falls_back(self) -> None:
        state: dict[str, Any] = {"current_transcript": "hey"}
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            return_value="casual",  # string instead of dict
        ):
            with patch("agents.routing.agent.asyncio.sleep", new_callable=AsyncMock):
                result = await _run_agent(state)
        assert result["route"] == RouteType.CASUAL.value
        assert result["routing_used_fallback"] is True

    @pytest.mark.asyncio
    async def test_extra_llm_fields_ignored(self) -> None:
        """LLM adding unexpected fields should not cause a failure."""
        state: dict[str, Any] = {"current_transcript": "set a reminder"}
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            return_value={
                "route": "task",
                "confidence": 0.91,
                "reasoning": "user wants a reminder",  # extra field
                "other_stuff": 123,
            },
        ):
            result = await _run_agent(state)
        assert result["route"] == "task"


# ---------------------------------------------------------------------------
# Route string written as plain string (not enum object)
# ---------------------------------------------------------------------------


class TestStateOutputFormat:
    @pytest.mark.asyncio
    async def test_route_written_as_plain_string(self) -> None:
        """state['route'] must be a str like 'casual', not RouteType.CASUAL."""
        state: dict[str, Any] = {"current_transcript": "hey"}
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            return_value=_make_llm_response("casual", 0.9),
        ):
            result = await _run_agent(state)

        assert isinstance(result["route"], str)
        assert result["route"] == "casual"  # plain string, not enum

    @pytest.mark.asyncio
    async def test_routing_confidence_is_float(self) -> None:
        state: dict[str, Any] = {"current_transcript": "hey"}
        with patch(
            "agents.routing.agent.generate_json",
            new_callable=AsyncMock,
            return_value=_make_llm_response("casual", 0.87),
        ):
            result = await _run_agent(state)
        assert isinstance(result["routing_confidence"], float)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------


def test_routing_agent_singleton_is_named() -> None:
    """The module-level routing_agent singleton must have the correct name."""
    assert routing_agent.name == "routing_agent"
