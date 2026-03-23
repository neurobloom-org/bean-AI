"""Tests for agents/therapeutic_convo/agent.py

Covers:
  - TherapeuticConvoAgent._run_safety_check
  - TherapeuticConvoAgent._retrieve_rag_context (timeout + error fallback)
  - TherapeuticConvoAgent._generate (LLM failure → fallback response)
  - TherapeuticConvoAgent._run_async_impl (empty transcript fast-exit, full pipeline)
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.therapeutic_convo.agent import (
    TherapeuticConvoAgent,
    _FALLBACK_RAG_CONTEXT,
    _FALLBACK_RESPONSE,
    therapy_agent,
)


def _make_ctx(state: dict[str, Any]) -> MagicMock:
    ctx = MagicMock()
    ctx.session.state = state
    return ctx


async def _run_agent(state: dict[str, Any]) -> dict[str, Any]:
    agent = TherapeuticConvoAgent(name="test_therapy")
    async for event in agent._run_async_impl(_make_ctx(state)):
        if event.actions and event.actions.state_delta:
            state.update(event.actions.state_delta)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# _run_safety_check
# ─────────────────────────────────────────────────────────────────────────────


class TestRunSafetyCheck:
    def setup_method(self):
        self.agent = TherapeuticConvoAgent(name="test_therapy")

    def test_clean_text_returns_empty_flag(self):
        delta = self.agent._run_safety_check("I feel a bit sad today", False)
        assert delta["safety_flag"] == ""
        assert delta["safety_matched_keywords"] == []

    def test_crisis_keyword_sets_flag(self):
        delta = self.agent._run_safety_check("I want to kill myself", False)
        assert delta["safety_flag"] == "crisis_keyword"
        assert len(delta["safety_matched_keywords"]) > 0

    def test_explicit_statement_sets_flag(self):
        delta = self.agent._run_safety_check("Tonight I will end it", False)
        assert delta["safety_flag"] == "explicit_statement"

    def test_explicit_takes_priority_over_crisis(self):
        text = "I already cut myself and tonight I will do it again"
        delta = self.agent._run_safety_check(text, False)
        assert delta["safety_flag"] == "explicit_statement"


# ─────────────────────────────────────────────────────────────────────────────
# _retrieve_rag_context
# ─────────────────────────────────────────────────────────────────────────────


class TestRetrieveRagContext:
    @pytest.mark.asyncio
    async def test_successful_retrieval(self):
        agent = TherapeuticConvoAgent(name="test_therapy")
        fake_techniques = [{"name": "Active Listening", "description": "Reflect feelings"}]

        with patch("agents.therapeutic_convo.agent.retrieve_cbt_techniques", new_callable=AsyncMock, return_value=fake_techniques):
            with patch("agents.therapeutic_convo.agent.format_techniques_for_prompt", return_value="Use active listening"):
                context, techniques = await agent._retrieve_rag_context("I feel sad", "sad")

        assert "active listening" in context.lower()
        assert techniques == fake_techniques

    @pytest.mark.asyncio
    async def test_timeout_returns_fallback(self):
        agent = TherapeuticConvoAgent(name="test_therapy")

        with patch("agents.therapeutic_convo.agent.retrieve_cbt_techniques", new_callable=AsyncMock, side_effect=asyncio.TimeoutError()):
            context, techniques = await agent._retrieve_rag_context("I feel sad", "sad")

        assert context == _FALLBACK_RAG_CONTEXT
        assert techniques == []

    @pytest.mark.asyncio
    async def test_exception_returns_fallback(self):
        agent = TherapeuticConvoAgent(name="test_therapy")

        with patch("agents.therapeutic_convo.agent.retrieve_cbt_techniques", new_callable=AsyncMock, side_effect=RuntimeError("DB down")):
            context, techniques = await agent._retrieve_rag_context("I feel sad", "sad")

        assert context == _FALLBACK_RAG_CONTEXT
        assert techniques == []


# ─────────────────────────────────────────────────────────────────────────────
# _generate
# ─────────────────────────────────────────────────────────────────────────────


class TestGenerate:
    @pytest.mark.asyncio
    async def test_successful_generation(self):
        agent = TherapeuticConvoAgent(name="test_therapy")
        with patch("agents.therapeutic_convo.agent.llm_generate", new_callable=AsyncMock, return_value="I hear you."):
            result = await agent._generate(system="sys prompt", user_prompt="user text")
        assert result == "I hear you."

    @pytest.mark.asyncio
    async def test_llm_error_returns_fallback(self):
        from shared.exceptions import LLMError
        agent = TherapeuticConvoAgent(name="test_therapy")
        with patch("agents.therapeutic_convo.agent.llm_generate", new_callable=AsyncMock, side_effect=LLMError("LLM failed")):
            result = await agent._generate(system="sys", user_prompt="user")
        assert result == _FALLBACK_RESPONSE

    @pytest.mark.asyncio
    async def test_empty_llm_response_returns_fallback(self):
        agent = TherapeuticConvoAgent(name="test_therapy")
        with patch("agents.therapeutic_convo.agent.llm_generate", new_callable=AsyncMock, return_value="   "):
            result = await agent._generate(system="sys", user_prompt="user")
        assert result == _FALLBACK_RESPONSE


# ─────────────────────────────────────────────────────────────────────────────
# _run_async_impl
# ─────────────────────────────────────────────────────────────────────────────


class TestTherapeuticAgentImpl:
    @pytest.mark.asyncio
    async def test_empty_transcript_returns_default_response(self):
        state: dict[str, Any] = {
            "current_transcript": "",
            "current_emotion": "neutral",
        }
        result = await _run_agent(state)
        assert "response_text" in result
        assert len(result["response_text"]) > 0

    @pytest.mark.asyncio
    async def test_full_pipeline_success(self):
        state: dict[str, Any] = {
            "current_transcript": "I feel really anxious about my exams",
            "current_emotion": "fearful",
            "memory_context": "User is 16 years old",
            "is_minor": True,
        }

        with patch("agents.therapeutic_convo.agent.retrieve_cbt_techniques", new_callable=AsyncMock, return_value=[]):
            with patch("agents.therapeutic_convo.agent.format_techniques_for_prompt", return_value="Use validation"):
                with patch("agents.therapeutic_convo.agent.llm_generate", new_callable=AsyncMock, return_value="I can hear how stressed you are."):
                    result = await _run_agent(state)

        assert result["response_text"] == "I can hear how stressed you are."
        assert "safety_flag" in result

    @pytest.mark.asyncio
    async def test_crisis_keyword_sets_safety_flag_in_state(self):
        state: dict[str, Any] = {
            "current_transcript": "I want to kill myself",
            "current_emotion": "sad",
            "is_minor": False,
        }

        with patch("agents.therapeutic_convo.agent.retrieve_cbt_techniques", new_callable=AsyncMock, return_value=[]):
            with patch("agents.therapeutic_convo.agent.format_techniques_for_prompt", return_value=""):
                with patch("agents.therapeutic_convo.agent.llm_generate", new_callable=AsyncMock, return_value="I'm here with you."):
                    result = await _run_agent(state)

        assert result["safety_flag"] in ("crisis_keyword", "explicit_statement")


def test_therapy_agent_singleton_name():
    assert therapy_agent.name == "therapy_agent"
