"""BEAN AI v5 — Pydantic schemas.

All datetimes are timezone-aware (UTC).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

from uuid import UUID


# ── Timezone-aware datetime alias ─────────────────────────────────────────────
UtcDatetime = Annotated[datetime, Field(default_factory=lambda: datetime.now(UTC))]


def utcnow() -> datetime:
    return datetime.now(UTC)


# ─────────────────────────────────────────────────────────────────────────────
# Auth
# ─────────────────────────────────────────────────────────────────────────────

class TokenPayload(BaseModel):
    sub: str
    email: str | None = None
    exp: int | None = None
    aud: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# User profile
# ─────────────────────────────────────────────────────────────────────────────

class UserProfile(BaseModel):
    user_id: str
    display_name: str | None = None
    preferred_name: str | None = None
    interests: list[str] = Field(default_factory=list)
    important_people: list[str] = Field(default_factory=list)
    personality_notes: list[str] = Field(default_factory=list)
    diagnosis_tags: list[str] = Field(default_factory=list)
    last_updated: UtcDatetime = Field(default_factory=utcnow)

    def to_context_string(self) -> str:
        parts: list[str] = []
        name = self.preferred_name or self.display_name
        if name:
            parts.append(f"Name: {name}")
        if self.interests:
            parts.append(f"Interests: {', '.join(self.interests[:8])}")
        if self.important_people:
            parts.append(f"Important people: {', '.join(self.important_people[:6])}")
        if self.personality_notes:
            parts.append(f"Notes: {'; '.join(self.personality_notes[:4])}")
        return "\n".join(parts) if parts else "No profile yet."


# ─────────────────────────────────────────────────────────────────────────────
# Memory types (used by MemoryAgent)
# ─────────────────────────────────────────────────────────────────────────────

class WorkingMemoryEntry(BaseModel):
    """A single turn in working memory (recent conversation)."""
    speaker: str          # "user" | "assistant"
    text: str
    timestamp: UtcDatetime = Field(default_factory=utcnow)


class EpisodicMemoryResult(BaseModel):
    """Result of a pgvector similarity search (metadata only, no source text)."""
    memory_id: str
    emotion_label: str | None = None
    similarity_score: float
    created_at: datetime


class MemoryContext(BaseModel):
    """Assembled memory context for injection into LLM prompts."""
    working_memory: list[WorkingMemoryEntry] = Field(default_factory=list)
    user_profile: UserProfile | None = None
    episodic_memories: list[EpisodicMemoryResult] = Field(default_factory=list)

    def to_prompt_string(self) -> str:
        parts: list[str] = []

        if self.user_profile:
            profile_str = self.user_profile.to_context_string()
            if profile_str != "No profile yet.":
                parts.append(f"User profile:\n{profile_str}")

        if self.working_memory:
            turns = []
            for entry in self.working_memory[-6:]:
                speaker = "User" if entry.speaker == "user" else "BEAN"
                turns.append(f"{speaker}: {entry.text}")
            parts.append("Recent conversation:\n" + "\n".join(turns))

        if self.episodic_memories:
            parts.append(
                f"Relevant past context: {len(self.episodic_memories)} similar interactions found."
            )

        return "\n\n".join(parts) if parts else "No memory context yet."


# ─────────────────────────────────────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────────────────────────────────────

class SessionCreate(BaseModel):
    user_id: str
    device_id: str | None = None


class SessionState(BaseModel):
    session_id: str
    user_id: str
    current_transcript: str = ""
    response_text: str = ""
    current_emotion: str = "neutral"
    route: str = "casual"
    turn_count: int = 0
    route_distribution: dict[str, int] = Field(default_factory=dict)
    memory_write_done: str = "pending"
    alert_triggered: bool = False
    started_at: UtcDatetime = Field(default_factory=utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket messages (ESP32 ↔ Cloud Run)
# ─────────────────────────────────────────────────────────────────────────────

class AudioFrameMessage(BaseModel):
    type: Literal["audio_frame"] = "audio_frame"
    audio_data: bytes
    sample_rate: int = 16000
    channels: int = 1


class TranscriptResult(BaseModel):
    type: Literal["transcript_partial", "transcript_final"]
    text: str
    confidence: float = 0.0
    is_final: bool = False


class EmotionResult(BaseModel):
    type: Literal["emotion_result"] = "emotion_result"
    emotion: str
    confidence: float

    @field_validator("confidence")
    @classmethod
    def round_confidence(cls, v: float) -> float:
        return round(v, 2)


class ResponseTextMessage(BaseModel):
    type: Literal["response_text"] = "response_text"
    text: str
    route: str
    turn_id: str


class ResponseAudioMessage(BaseModel):
    type: Literal["response_audio"] = "response_audio"
    audio_chunk: bytes
    is_final: bool = False
    turn_id: str


class FillerPhraseResult(BaseModel):
    """Result of filler phrase selection (active listen agent)."""
    phrase: str
    audio_cache_key: str
    audio_b64: str | None = None     # pre-cached TTS audio if available


class FillerAudioMessage(BaseModel):
    type: Literal["filler_audio"] = "filler_audio"
    filler_text: str


class MusicCommand(BaseModel):
    """Music command sent to ESP32 hardware."""
    action: str                      # MusicAction enum value
    genre_folder: str | None = None
    shuffle: bool = True
    volume: int | None = None


class MusicControlMessage(BaseModel):
    type: Literal[
        "play_music", "stop_music", "next_track",
        "skip_track", "pause_music", "resume_music", "set_volume"
    ]
    genre_folder: str | None = None
    shuffle: bool | None = None
    volume: int | None = None


class AlertMessage(BaseModel):
    type: Literal["alert"] = "alert"
    alert_level: str
    message: str


class RobotStatusMessage(BaseModel):
    type: Literal["robot_status"] = "robot_status"
    battery_level: int | None = None
    wifi_rssi: int | None = None
    firmware_version: str | None = None


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str


class PingMessage(BaseModel):
    type: Literal["ping"] = "ping"
    timestamp: UtcDatetime = Field(default_factory=utcnow)


class PongMessage(BaseModel):
    type: Literal["pong"] = "pong"
    timestamp: UtcDatetime = Field(default_factory=utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Alert
# ─────────────────────────────────────────────────────────────────────────────

class AlertCreate(BaseModel):
    user_id: str
    session_id: str | None = None
    alert_level: str
    alert_factors: list[str]


class AlertResponse(BaseModel):
    id: str
    user_id: str
    alert_level: str
    alert_factors: list[str]
    notified_guardian: bool
    sms_sent: bool
    acknowledged: bool
    created_at: UtcDatetime


# ─────────────────────────────────────────────────────────────────────────────
# Task / reminder
# ─────────────────────────────────────────────────────────────────────────────

class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    description: str | None = Field(None, max_length=1000)
    due_at: datetime | None = None


class TaskResponse(BaseModel):
    id: str
    user_id: str
    title: str
    description: str | None
    due_at: datetime | None
    status: str
    created_at: UtcDatetime
    updated_at: UtcDatetime


# ─────────────────────────────────────────────────────────────────────────────
# Emotion graph data
# ─────────────────────────────────────────────────────────────────────────────

class DailyEmotionSummary(BaseModel):
    day: datetime
    emotion: str
    count: int
    avg_confidence: float


class WeeklySessionActivity(BaseModel):
    week: datetime
    session_count: int
    total_duration_seconds: int
    avg_turns_per_session: float
    most_common_emotion: str | None


# ─────────────────────────────────────────────────────────────────────────────
# LLM routing output
# ─────────────────────────────────────────────────────────────────────────────

class RoutingDecision(BaseModel):
    route: Literal["casual", "therapy", "task", "music", "alert"]
    confidence: float
    reasoning: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    supabase: bool
    deepgram_configured: bool
    elevenlabs_configured: bool
    gemini_configured: bool
    checked_at: UtcDatetime = Field(default_factory=utcnow)


# ─────────────────────────────────────────────────────────────────────────────
# Guardian / doctor API
# ─────────────────────────────────────────────────────────────────────────────
class GuardianLinkCreate(BaseModel):
    guardian_user_id: UUID
    relationship: Literal["guardian", "doctor", "parent", "caregiver"] = "guardian"

class GuardianPatientOverview(BaseModel):
    patient_user_id: str
    display_name: str | None
    diagnosis_tags: list[str]
    recent_emotion_trend: list[DailyEmotionSummary]
    recent_alerts: list[AlertResponse]
    last_session_at: datetime | None
    total_sessions_this_week: int


# ─────────────────────────────────────────────────────────────────────────────
# Internal
# ─────────────────────────────────────────────────────────────────────────────

class EpisodicsEmbedRequest(BaseModel):
    session_id: str
    user_id: str
    user_text: str
    assistant_text: str
    emotion: str


class PurgeResult(BaseModel):
    transcripts: str
    emotions: str
    memories: str
    sessions: str
    rate_limits: str
    ran_at: UtcDatetime = Field(default_factory=utcnow)
