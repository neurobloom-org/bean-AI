"""Tests for shared/enums.py"""

from __future__ import annotations

import pytest

from shared.enums import AlertFactor, AlertLevel, EmotionLabel, RouteType


class TestEmotionLabel:
    def test_negative_set_contains_expected(self):
        neg = EmotionLabel.negative()
        assert EmotionLabel.SAD in neg
        assert EmotionLabel.ANGRY in neg
        assert EmotionLabel.FEARFUL in neg
        assert EmotionLabel.DISGUSTED in neg

    def test_positive_set_contains_expected(self):
        pos = EmotionLabel.positive()
        assert EmotionLabel.HAPPY in pos
        assert EmotionLabel.CALM in pos

    def test_negative_and_positive_are_disjoint(self):
        assert EmotionLabel.negative().isdisjoint(EmotionLabel.positive())

    def test_neutral_in_neither_set(self):
        assert EmotionLabel.NEUTRAL not in EmotionLabel.negative()
        assert EmotionLabel.NEUTRAL not in EmotionLabel.positive()

    def test_string_value_construction(self):
        assert EmotionLabel("sad") == EmotionLabel.SAD

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            EmotionLabel("depressed")


class TestRouteType:
    def test_therapy_uses_pro_llm(self):
        assert RouteType.THERAPY.uses_pro_llm() is True

    def test_alert_uses_pro_llm(self):
        assert RouteType.ALERT.uses_pro_llm() is True

    def test_casual_does_not_use_pro_llm(self):
        assert RouteType.CASUAL.uses_pro_llm() is False

    def test_task_does_not_use_pro_llm(self):
        assert RouteType.TASK.uses_pro_llm() is False

    def test_music_does_not_use_pro_llm(self):
        assert RouteType.MUSIC.uses_pro_llm() is False

    def test_string_value_construction(self):
        assert RouteType("casual") == RouteType.CASUAL


class TestAlertLevel:
    def test_high_requires_sms(self):
        assert AlertLevel.HIGH.requires_sms() is True

    def test_crisis_requires_sms(self):
        assert AlertLevel.CRISIS.requires_sms() is True

    def test_none_does_not_require_sms(self):
        assert AlertLevel.NONE.requires_sms() is False

    def test_low_does_not_require_sms(self):
        assert AlertLevel.LOW.requires_sms() is False

    def test_medium_does_not_require_sms(self):
        assert AlertLevel.MEDIUM.requires_sms() is False

    def test_scores_ordered(self):
        assert AlertLevel.NONE.score() < AlertLevel.LOW.score()
        assert AlertLevel.LOW.score() < AlertLevel.MEDIUM.score()
        assert AlertLevel.MEDIUM.score() < AlertLevel.HIGH.score()
        assert AlertLevel.HIGH.score() < AlertLevel.CRISIS.score()

    def test_string_value_construction(self):
        assert AlertLevel("crisis") == AlertLevel.CRISIS


class TestAlertFactor:
    def test_all_factors_have_values(self):
        factors = [
            AlertFactor.F1_CRISIS_KEYWORD,
            AlertFactor.F2_NEGATIVE_EMOTION,
            AlertFactor.F3_ESCALATION,
            AlertFactor.F4_VULNERABILITY,
            AlertFactor.F5_EXPLICIT_STATEMENT,
        ]
        for f in factors:
            assert isinstance(f.value, str)
            assert len(f.value) > 0

    def test_factor_values_are_unique(self):
        values = [f.value for f in AlertFactor]
        assert len(values) == len(set(values))
