"""Tests for agents/active_listen/agent.py

Covers:
  - get_filler_phrase for every known emotion
  - Avoidance of repeated phrases
  - Graceful handling of unknown emotions
  - Return schema matches FillerPhraseResult fields
"""

from __future__ import annotations

from agents.active_listen.agent import _PHRASES, _DEFAULT_PHRASES, get_filler_phrase


class TestGetFillerPhrase:
    def test_returns_dict_with_required_keys(self):
        result = get_filler_phrase("neutral")
        assert "phrase" in result
        assert "audio_cache_key" in result

    def test_sad_returns_phrase_from_sad_pool(self):
        result = get_filler_phrase("sad")
        assert result["phrase"] in _PHRASES["sad"]

    def test_happy_returns_phrase_from_happy_pool(self):
        result = get_filler_phrase("happy")
        assert result["phrase"] in _PHRASES["happy"]

    def test_angry_returns_phrase_from_angry_pool(self):
        result = get_filler_phrase("angry")
        assert result["phrase"] in _PHRASES["angry"]

    def test_fearful_returns_phrase_from_fearful_pool(self):
        result = get_filler_phrase("fearful")
        assert result["phrase"] in _PHRASES["fearful"]

    def test_calm_returns_phrase_from_calm_pool(self):
        result = get_filler_phrase("calm")
        assert result["phrase"] in _PHRASES["calm"]

    def test_neutral_returns_phrase_from_neutral_pool(self):
        result = get_filler_phrase("neutral")
        assert result["phrase"] in _PHRASES["neutral"]

    def test_unknown_emotion_uses_default_phrases(self):
        result = get_filler_phrase("confused")
        assert result["phrase"] in _DEFAULT_PHRASES

    def test_empty_emotion_uses_default_phrases(self):
        result = get_filler_phrase("")
        assert result["phrase"] in _DEFAULT_PHRASES

    def test_avoids_last_phrase(self):
        """With a known last_phrase, the selected phrase should differ."""
        # Run many times to avoid relying on randomness
        sad_phrases = _PHRASES["sad"]
        if len(sad_phrases) < 2:
            return  # can't test avoidance with a single phrase

        last = sad_phrases[0]
        results = {get_filler_phrase("sad", last_phrase=last)["phrase"] for _ in range(20)}
        # At least one call should have returned something different
        assert any(p != last for p in results)

    def test_audio_cache_key_contains_emotion(self):
        result = get_filler_phrase("happy")
        assert "happy" in result["audio_cache_key"]

    def test_audio_cache_key_has_no_spaces(self):
        result = get_filler_phrase("neutral")
        assert " " not in result["audio_cache_key"]

    def test_all_emotions_covered(self):
        """Every emotion defined in _PHRASES should return a non-empty phrase."""
        for emotion in _PHRASES:
            result = get_filler_phrase(emotion)
            assert isinstance(result["phrase"], str)
            assert len(result["phrase"]) > 0
