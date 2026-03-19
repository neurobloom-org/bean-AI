"""
shared/enums.py
----------------
Enumerations used across the Bean AI system.
"""

from enum import Enum, StrEnum


class Framework(StrEnum):
    """Therapy frameworks used by Bean."""
    CBT  = "cbt"
    MBCT = "mbct"
    DCT  = "dct"


class MoodScore(int, Enum):
    """Mood scores from 1 (very low) to 5 (positive)."""
    VERY_LOW  = 1
    LOW       = 2
    NEUTRAL   = 3
    IMPROVING = 4
    POSITIVE  = 5


class CrisisLevel(int, Enum):
    """Crisis detection levels."""
    NONE     = 0
    ELEVATED = 1
    CRITICAL = 2


class AgentType(StrEnum):
    """Types of agents in Bean AI system."""
    CASUAL_CHAT   = "casual_chat"
    THERAPY       = "therapy"
    TASK          = "task"
    MUSIC         = "music"
    MEMORY        = "memory"
    TTS           = "tts"
    STT           = "stt"
    ACTIVE_LISTEN = "active_listen"
    ALERT         = "alert"


class SessionStatus(StrEnum):
    """Status of a Bean session."""
    ACTIVE    = "active"
    COMPLETED = "completed"
    ESCALATED = "escalated"

 