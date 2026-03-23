"""Tests for agents/memory_writer/agent.py

Covers:
  - _clean_optional_string
  - _coerce_string_list
  - _merge_string_lists
  - _sanitize_emotion
  - _safe_redact_and_truncate
  - _is_valid_embedding
  - _parse_extracted_facts
  - ExtractedFacts Pydantic validation
  - MemoryWriterAgent._build_merged_profile_payload
  - MemoryWriterAgent._run_async_impl (mocked DB + LLM + embedding)

No real external calls.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.memory_writer.agent import (
    ExtractedFacts,
    MemoryWriterAgent,
    _clean_optional_string,
    _coerce_string_list,
    _is_valid_embedding,
    _merge_string_lists,
    _parse_extracted_facts,
    _sanitize_emotion,
    _safe_redact_and_truncate,
    memory_writer_agent,
)


# ─────────────────────────────────────────────────────────────────────────────
# _clean_optional_string
# ─────────────────────────────────────────────────────────────────────────────


class TestCleanOptionalString:
    def test_normal_string_cleaned(self):
        assert _clean_optional_string("  hello world  ") == "hello world"

    def test_collapses_internal_whitespace(self):
        assert _clean_optional_string("hello   world") == "hello world"

    def test_truncates_to_max_len(self):
        result = _clean_optional_string("a" * 200, max_len=10)
        assert len(result) == 10

    def test_empty_string_returns_none(self):
        assert _clean_optional_string("") is None

    def test_whitespace_only_returns_none(self):
        assert _clean_optional_string("   ") is None

    def test_non_string_returns_none(self):
        assert _clean_optional_string(None) is None
        assert _clean_optional_string(42) is None
        assert _clean_optional_string([]) is None


# ─────────────────────────────────────────────────────────────────────────────
# _coerce_string_list
# ─────────────────────────────────────────────────────────────────────────────


class TestCoerceStringList:
    def test_valid_list_of_strings(self):
        result = _coerce_string_list(["football", "music", "art"])
        assert result == ["football", "music", "art"]

    def test_deduplicates_case_insensitively(self):
        result = _coerce_string_list(["Music", "music", "MUSIC"])
        assert len(result) == 1

    def test_strips_whitespace_from_items(self):
        result = _coerce_string_list(["  hello  "])
        assert result == ["hello"]

    def test_skips_empty_items(self):
        result = _coerce_string_list(["valid", "", "  "])
        assert result == ["valid"]

    def test_respects_max_items(self):
        result = _coerce_string_list(["item"] * 20, max_items=5)
        assert len(result) == 1  # dedup removes all but one

    def test_non_list_returns_empty(self):
        assert _coerce_string_list(None) == []
        assert _coerce_string_list("string") == []
        assert _coerce_string_list(42) == []

    def test_truncates_long_items(self):
        long_item = "a" * 200
        result = _coerce_string_list([long_item], max_item_len=10)
        assert len(result[0]) == 10


# ─────────────────────────────────────────────────────────────────────────────
# _merge_string_lists
# ─────────────────────────────────────────────────────────────────────────────


class TestMergeStringLists:
    def test_appends_new_items(self):
        result = _merge_string_lists(["existing"], ["new"])
        assert result == ["existing", "new"]

    def test_deduplicates_case_insensitively(self):
        result = _merge_string_lists(["Football"], ["football"])
        assert len(result) == 1
        assert result[0] == "Football"  # original casing preserved

    def test_existing_items_come_first(self):
        result = _merge_string_lists(["b", "a"], ["c"])
        assert result == ["b", "a", "c"]

    def test_empty_existing(self):
        result = _merge_string_lists([], ["new"])
        assert result == ["new"]

    def test_empty_new_items(self):
        result = _merge_string_lists(["existing"], [])
        assert result == ["existing"]


# ─────────────────────────────────────────────────────────────────────────────
# _sanitize_emotion
# ─────────────────────────────────────────────────────────────────────────────


class TestSanitizeEmotion:
    def test_valid_emotion_passthrough(self):
        assert _sanitize_emotion("sad") == "sad"

    def test_strips_special_chars(self):
        result = _sanitize_emotion("sad!@#")
        assert "!" not in result
        assert "@" not in result

    def test_truncates_long_emotion(self):
        result = _sanitize_emotion("a" * 100)
        assert len(result) <= 40

    def test_empty_string_returns_neutral(self):
        assert _sanitize_emotion("") == "neutral"

    def test_uppercase_lowercased(self):
        assert _sanitize_emotion("SAD") == "sad"


# ─────────────────────────────────────────────────────────────────────────────
# _safe_redact_and_truncate
# ─────────────────────────────────────────────────────────────────────────────


class TestSafeRedactAndTruncate:
    def test_truncates_to_max_chars(self):
        long_text = "a" * 600
        result = _safe_redact_and_truncate(long_text, 500)
        assert len(result) <= 500

    def test_redacts_phone_numbers(self):
        text = "Call me at +94771234567 anytime"
        result = _safe_redact_and_truncate(text, 500)
        assert "+94771234567" not in result

    def test_redacts_email_addresses(self):
        text = "Email me at user@example.com please"
        result = _safe_redact_and_truncate(text, 500)
        assert "user@example.com" not in result

    def test_empty_string_returns_empty(self):
        result = _safe_redact_and_truncate("", 500)
        assert result == ""


# ─────────────────────────────────────────────────────────────────────────────
# _is_valid_embedding
# ─────────────────────────────────────────────────────────────────────────────


class TestIsValidEmbedding:
    def test_valid_1536_float_list(self):
        embedding = [0.1] * 1536
        assert _is_valid_embedding(embedding, expected_dimensions=1536) is True

    def test_wrong_length_returns_false(self):
        assert _is_valid_embedding([0.1] * 512, expected_dimensions=1536) is False

    def test_non_list_returns_false(self):
        assert _is_valid_embedding("not a list", expected_dimensions=1536) is False
        assert _is_valid_embedding(None, expected_dimensions=1536) is False

    def test_list_with_non_numeric_returns_false(self):
        bad = ["string"] + [0.1] * 1535
        assert _is_valid_embedding(bad, expected_dimensions=1536) is False


# ─────────────────────────────────────────────────────────────────────────────
# _parse_extracted_facts
# ─────────────────────────────────────────────────────────────────────────────


class TestParseExtractedFacts:
    def test_parses_dict(self):
        raw = {"name": "Isara", "new_interests": ["chess"]}
        facts = _parse_extracted_facts(raw)
        assert facts.name == "Isara"
        assert facts.new_interests == ["chess"]

    def test_parses_json_string(self):
        raw = '{"name": "Alex", "new_interests": ["painting"]}'
        facts = _parse_extracted_facts(raw)
        assert facts.name == "Alex"

    def test_extra_fields_ignored(self):
        raw = {"name": "Isara", "unknown_field": "ignored", "new_interests": []}
        facts = _parse_extracted_facts(raw)
        assert facts.name == "Isara"

    def test_empty_dict_gives_defaults(self):
        facts = _parse_extracted_facts({})
        assert facts.name is None
        assert facts.new_interests == []


# ─────────────────────────────────────────────────────────────────────────────
# ExtractedFacts Pydantic validation
# ─────────────────────────────────────────────────────────────────────────────


class TestExtractedFactsValidation:
    def test_name_truncated_to_120(self):
        facts = ExtractedFacts(name="a" * 200)
        assert len(facts.name) == 120

    def test_non_string_name_becomes_none(self):
        facts = ExtractedFacts(name=42)
        assert facts.name is None

    def test_interests_limited_to_10(self):
        facts = ExtractedFacts(new_interests=[f"interest_{i}" for i in range(20)])
        assert len(facts.new_interests) <= 10

    def test_significant_event_truncated_to_180(self):
        facts = ExtractedFacts(significant_event="x" * 300)
        assert len(facts.significant_event) == 180

    def test_non_list_interests_become_empty(self):
        facts = ExtractedFacts(new_interests="not a list")
        assert facts.new_interests == []


# ─────────────────────────────────────────────────────────────────────────────
# MemoryWriterAgent._build_merged_profile_payload
# ─────────────────────────────────────────────────────────────────────────────


class TestBuildMergedProfilePayload:
    def setup_method(self):
        self.agent = MemoryWriterAgent(name="test_mw")

    def test_merges_interests_with_existing(self):
        facts = ExtractedFacts(new_interests=["chess"])
        existing = {"interests": ["football"]}
        payload = self.agent._build_merged_profile_payload(facts=facts, existing=existing)
        assert "chess" in payload["interests"]
        assert "football" in payload["interests"]

    def test_does_not_overwrite_name_with_none(self):
        facts = ExtractedFacts(name=None)
        existing = {"display_name": "Isara"}
        payload = self.agent._build_merged_profile_payload(facts=facts, existing=existing)
        assert "display_name" not in payload

    def test_sets_name_when_present(self):
        facts = ExtractedFacts(name="Alex")
        payload = self.agent._build_merged_profile_payload(facts=facts, existing={})
        assert payload["display_name"] == "Alex"

    def test_empty_facts_return_empty_payload(self):
        facts = ExtractedFacts()
        payload = self.agent._build_merged_profile_payload(facts=facts, existing={})
        assert payload == {}

    def test_significant_event_added_to_personality_notes(self):
        facts = ExtractedFacts(significant_event="passed driving test")
        payload = self.agent._build_merged_profile_payload(facts=facts, existing={})
        assert any("passed driving test" in note for note in payload["personality_notes"])

    def test_payload_contains_last_updated_when_non_empty(self):
        facts = ExtractedFacts(name="Alex")
        payload = self.agent._build_merged_profile_payload(facts=facts, existing={})
        assert "last_updated" in payload


# ─────────────────────────────────────────────────────────────────────────────
# MemoryWriterAgent._run_async_impl — fast exits
# ─────────────────────────────────────────────────────────────────────────────


def _make_ctx(state: dict[str, Any]) -> MagicMock:
    ctx = MagicMock()
    ctx.session.state = state
    return ctx


async def _run_agent(state: dict[str, Any]) -> dict[str, Any]:
    agent = MemoryWriterAgent(name="test_mw")
    async for event in agent._run_async_impl(_make_ctx(state)):
        if event.actions and event.actions.state_delta:
            state.update(event.actions.state_delta)
    return state


class TestMemoryWriterAgentFastExits:
    @pytest.mark.asyncio
    async def test_missing_user_id_skips(self):
        state: dict[str, Any] = {
            "current_transcript": "Hello",
            "response_text": "Hi",
        }
        result = await _run_agent(state)
        assert result["memory_write_done"] == "skipped"

    @pytest.mark.asyncio
    async def test_empty_turn_skips(self):
        state: dict[str, Any] = {
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_transcript": "",
            "response_text": "",
        }
        result = await _run_agent(state)
        assert result["memory_write_done"] == "skipped"


class TestMemoryWriterAgentSuccess:
    @pytest.mark.asyncio
    async def test_successful_run_sets_true(self):
        state: dict[str, Any] = {
            "user_id": "user-1",
            "session_id": "sess-1",
            "current_transcript": "I love playing chess",
            "response_text": "That's great!",
            "current_emotion": "happy",
        }

        fake_facts = {
            "name": None,
            "preferred_name": None,
            "new_interests": ["chess"],
            "new_important_people": [],
            "new_personality_notes": [],
            "significant_event": None,
        }
        fake_embedding = [0.1] * 1536

        # Build a fully synchronous mock chain so awaiting get_service_client()
        # returns a plain MagicMock whose .table().select()... chain is also sync.
        db = MagicMock()
        db.table.return_value.select.return_value.eq.return_value.maybe_single.return_value.execute = AsyncMock(
            return_value=MagicMock(data=None)
        )
        db.table.return_value.upsert.return_value.execute = AsyncMock(
            return_value=MagicMock(data=[{"user_id": "user-1"}])
        )
        db.table.return_value.insert.return_value.execute = AsyncMock(
            return_value=MagicMock(data=[{"id": "mem-1"}])
        )

        async def fake_get_client():
            return db

        with patch("agents.memory_writer.agent.generate_json", new_callable=AsyncMock, return_value=fake_facts):
            with patch("agents.memory_writer.agent.get_embedding", new_callable=AsyncMock, return_value=fake_embedding):
                with patch("agents.memory_writer.agent.get_service_client", side_effect=fake_get_client):
                    result = await _run_agent(state)

        assert result["memory_write_done"] == "true"


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────


def test_memory_writer_agent_singleton_name():
    assert memory_writer_agent.name == "memory_writer_agent"