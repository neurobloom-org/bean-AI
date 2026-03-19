"""BEAN AI v5 — Enums."""

from enum import Enum


class EmotionLabel(str, Enum):
    HAPPY      = "happy"
    SAD        = "sad"
    ANGRY      = "angry"
    FEARFUL    = "fearful"
    DISGUSTED  = "disgusted"
    SURPRISED  = "surprised"
    NEUTRAL    = "neutral"
    CALM       = "calm"

    @classmethod
    def negative(cls) -> set["EmotionLabel"]:
        return {cls.SAD, cls.ANGRY, cls.FEARFUL, cls.DISGUSTED}

    @classmethod
    def positive(cls) -> set["EmotionLabel"]:
        return {cls.HAPPY, cls.CALM}


class RouteType(str, Enum):
    CASUAL  = "casual"
    THERAPY = "therapy"
    TASK    = "task"
    MUSIC   = "music"
    ALERT   = "alert"

    def uses_pro_llm(self) -> bool:
        return self in {RouteType.THERAPY, RouteType.ALERT}


class AlertLevel(str, Enum):
    NONE   = "none"
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"
    CRISIS = "crisis"

    def requires_sms(self) -> bool:
        return self in {AlertLevel.HIGH, AlertLevel.CRISIS}

    def score(self) -> int:
        return {"none": 0, "low": 1, "medium": 2, "high": 3, "crisis": 4}[self.value]


class AlertFactor(str, Enum):
    F1_CRISIS_KEYWORD     = "f1_crisis_keyword"
    F2_NEGATIVE_EMOTION   = "f2_negative_emotion"
    F3_ESCALATION         = "f3_escalation_pattern"
    F4_VULNERABILITY      = "f4_vulnerability"
    F5_EXPLICIT_STATEMENT = "f5_explicit_statement"


class SessionStatus(str, Enum):
    ACTIVE  = "active"
    PAUSED  = "paused"
    ENDED   = "ended"
    EXPIRED = "expired"


class TaskStatus(str, Enum):
    PENDING   = "pending"
    REMINDED  = "reminded"
    COMPLETED = "completed"
    SNOOZED   = "snoozed"
    CANCELLED = "cancelled"


class MusicAction(str, Enum):
    PLAY       = "play_music"
    STOP       = "stop_music"
    NEXT       = "next_track"
    SKIP       = "skip_track"
    PAUSE      = "pause_music"
    RESUME     = "resume_music"
    SET_VOLUME = "set_volume"
