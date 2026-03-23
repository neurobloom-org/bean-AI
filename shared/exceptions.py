"""BEAN AI v1 — Custom exceptions.

All application exceptions inherit from BEANError so callers can catch
the full hierarchy with a single except clause if needed.

Hierarchy:
    BEANError
    ├── ConfigError
    ├── AuthError
    │   ├── TokenExpiredError
    │   └── WebSocketAuthError
    ├── DeepgramConnectionError
    ├── ElevenLabsError
    ├── LLMError
    ├── EmbeddingError
    ├── SupabaseError
    ├── CalendarError
    ├── RAGError
    ├── TwilioError
    ├── CrisisDetectedError
    ├── SessionNotFoundError
    └── RateLimitExceededError

Usage note — WebSocketAuthError:
    The canonical definition is here. Do NOT define a local WebSocketAuthError
    in api/middleware/auth_middleware.py or anywhere else — import this one.
    Any code catching WebSocketAuthError must import from shared.exceptions.
"""

__all__ = [
    "BEANError",
    "ConfigError",
    "AuthError",
    "TokenExpiredError",
    "WebSocketAuthError",
    "DeepgramConnectionError",
    "ElevenLabsError",
    "LLMError",
    "EmbeddingError",
    "SupabaseError",
    "CalendarError",
    "RAGError",
    "TwilioError",
    "CrisisDetectedError",
    "SessionNotFoundError",
    "RateLimitExceededError",
]


class BEANError(Exception):
    """Base exception for all BEAN AI errors.

    Catch this to handle any application-level error without caring about
    the specific subsystem that failed.
    """


class ConfigError(BEANError):
    """Missing or invalid configuration.

    Raised at startup when a required setting is absent or has an
    invalid value (e.g. empty API key, out-of-range integer).
    """


# ── Auth ──────────────────────────────────────────────────────────────────────


class AuthError(BEANError):
    """Authentication or authorisation failure.

    Raised by auth_middleware when a JWT cannot be validated for any reason
    other than expiry. Callers should respond with HTTP 401.
    """


class TokenExpiredError(AuthError):
    """JWT has expired.

    Subclass of AuthError so callers that catch AuthError broadly still
    handle expiry correctly. Callers that need to distinguish expiry
    (to prompt re-authentication) from other failures can catch this
    specifically before catching AuthError.

    HTTP response: 401 with WWW-Authenticate: Bearer error="invalid_token"
    WebSocket response: close code 4401
    """


class WebSocketAuthError(AuthError):
    """WebSocket authentication failed.

    Raised by authenticate_websocket() after the WebSocket has already
    been closed with code 4401. Callers must NOT attempt to close the
    WebSocket again after catching this.

    This is the ONLY definition of WebSocketAuthError in the codebase.
    Do not define a local version in auth_middleware.py.
    """


# ── External services ─────────────────────────────────────────────────────────


class DeepgramConnectionError(BEANError):
    """Failed to connect to or communicate with Deepgram STT.

    Raised by deepgram_service when the WebSocket handshake fails,
    the connection drops permanently, or a keepalive times out.
    """


class ElevenLabsError(BEANError):
    """ElevenLabs TTS generation failure.

    Raised by elevenlabs_service when the API call fails or returns
    an empty audio stream.
    """


class LLMError(BEANError):
    """LLM generation failure.

    Raised by llm_service when the Gemini API call fails or returns
    an unusable response. Not raised for JSON parse failures — those
    surface as ValueError from generate_json().
    """


class EmbeddingError(BEANError):
    """Embedding generation failure.

    Raised by embedding_service when the OpenAI embeddings API call
    fails, or when the returned vector has unexpected dimensions.
    """


class SupabaseError(BEANError):
    """Supabase DB or Auth operation failure.

    Raised when a Supabase table operation fails after retries, or
    when an auth admin call (e.g. get_user_by_id) returns an error.
    """


class CalendarError(BEANError):
    """Google Calendar API failure.

    Raised by calendar_service when a Calendar REST call fails.
    Non-fatal for task creation — calendar sync is best-effort.
    """


class RAGError(BEANError):
    """RAG retrieval failure.

    Raised by rag_service when the pgvector similarity search fails.
    TherapeuticConvoAgent falls back to default techniques on RAGError.
    """


class TwilioError(BEANError):
    """Twilio SMS send failure.

    Raised by twilio_service.send_guardian_alert() when all retry
    attempts are exhausted. Not raised by send_sms() — that returns
    False on failure instead.
    """


# ── Safety ────────────────────────────────────────────────────────────────────


class CrisisDetectedError(BEANError):
    """Crisis-level safety event detected — triggers immediate alert dispatch.

    Raised by SafetyService.assess_turn() when alert_level == CRISIS.
    The orchestrator catches this and ensures the AlertAgent has already
    dispatched or will immediately dispatch a guardian SMS.
    """


# ── Session ───────────────────────────────────────────────────────────────────


class SessionNotFoundError(BEANError):
    """Session does not exist or has expired.

    Raised when the orchestrator or WebSocket handler cannot find an
    active session record for the given session_id.
    """


class RateLimitExceededError(BEANError):
    """Rate limit exceeded for this user or IP.

    Raised by rate_limiter when check_rate_limit() returns allowed=False.
    HTTP callers should respond with 429; WebSocket callers should drop
    the frame silently rather than closing the connection.
    """
