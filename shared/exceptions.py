"""BEAN AI v5 — Custom exceptions."""


class BEANError(Exception):
    """Base exception for all BEAN AI errors."""


class ConfigError(BEANError):
    """Missing or invalid configuration."""


# ── Auth ──────────────────────────────────────────────────────────────────────
class AuthError(BEANError):
    """Authentication or authorisation failure."""

class TokenExpiredError(AuthError):
    """JWT has expired."""

class WebSocketAuthError(AuthError):
    """WebSocket authentication failed."""


# ── External services ─────────────────────────────────────────────────────────
class DeepgramConnectionError(BEANError):
    """Failed to connect or communicate with Deepgram STT."""

class ElevenLabsError(BEANError):
    """ElevenLabs TTS failure."""

class LLMError(BEANError):
    """LLM generation failure."""

class EmbeddingError(BEANError):
    """Embedding generation failure."""

class SupabaseError(BEANError):
    """Supabase DB/Auth operation failure."""

class CalendarError(BEANError):
    """Google Calendar API failure."""

class RAGError(BEANError):
    """RAG retrieval failure."""


# ── Safety ────────────────────────────────────────────────────────────────────
class CrisisDetectedError(BEANError):
    """Crisis-level safety event detected — triggers immediate alert."""


# ── Session ───────────────────────────────────────────────────────────────────
class SessionNotFoundError(BEANError):
    """Session does not exist or has expired."""

class RateLimitExceededError(BEANError):
    """Rate limit exceeded for this user."""
