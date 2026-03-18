"""
shared/schemas.py
------------------
Pydantic schemas for Bean AI system.
Imports from enums and config — must be built last in shared/.
"""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from shared.enums import Framework, MoodScore, CrisisLevel, AgentType


# - Session Schemas -

class SessionCreate(BaseModel):
    """Schema for creating a new session."""
    user_id:    str
    user_text:  str


class SessionResponse(BaseModel):
    """Schema for session response sent back to ESP32."""
    transcript:     str
    response:       str
    mood_score:     MoodScore
    framework_used: Optional[Framework] = None
    crisis_level:   CrisisLevel
    agent_type:     AgentType


# - Mood Schemas ─

class MoodLog(BaseModel):
    """Schema for logging mood to Supabase."""
    user_id:     str
    mood_score:  MoodScore
    timestamp:   datetime


# ─ Crisis Schemas ─

class CrisisAlert(BaseModel):
    """Schema for crisis alert logging."""
    user_id:   str
    level:     CrisisLevel
    user_text: str
    notified:  bool = False


# ─ RAG Schemas ─

class RAGChunk(BaseModel):
    """Schema for a single RAG chunk."""
    text:      str
    framework: Framework
    topic:     str
    source:    str
    score:     Optional[float] = None


class RAGResponse(BaseModel):
    """Schema for RAG retrieval result."""
    chunks:         list[RAGChunk]
    framework_used: Framework