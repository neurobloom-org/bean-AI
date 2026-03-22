"""BEAN AI v1 — Privacy-First Configuration.

All settings are read from environment variables (or a .env file).
Field names map directly to env var names (case-insensitive).
For example, `supabase_url` reads from SUPABASE_URL.

Alert threshold semantics:
    alert_threshold        — factors required to trigger HIGH/CRISIS for adults (default 3)
    minor_alert_threshold  — factors required to trigger HIGH/CRISIS for minors (default 2)
    Minors have a lower threshold because F4 (vulnerability) is always pre-counted,
    so they effectively start with one factor already active.

    guardian_alert_cooldown_seconds — minimum seconds between guardian SMS alerts
    for the same user. Prevents spam on long sessions with sustained crisis signals.
    (Not yet implemented in AlertAgent — reserved for future use.)
"""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):  # type: ignore[misc]
    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "BEAN AI"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    log_level: str = "INFO"

    # ── FastAPI / Server ──────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8080
    cors_allowed_origins: str = "http://localhost:3000"

    # ── Supabase ──────────────────────────────────────────────────────────────
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""
    supabase_jwt_secret: str = ""

    # ── Tiered LLM — Google Gemini ────────────────────────────────────────────
    google_api_key: str | None = None
    llm_cheap_model: str = "gemini-2.0-flash"
    llm_cheap_max_tokens: int = 1024
    llm_cheap_temperature: float = 0.7
    llm_pro_model: str = "gemini-2.5-pro"
    llm_pro_max_tokens: int = 2048
    llm_pro_temperature: float = 0.8

    # ── OpenAI — Embeddings ───────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536

    # ── Deepgram — STT ────────────────────────────────────────────────────────
    deepgram_api_key: str = ""
    deepgram_model: str = "nova-2"
    deepgram_language: str = "en-US"
    deepgram_sample_rate: int = 16000
    deepgram_encoding: str = "linear16"
    deepgram_channels: int = 1
    deepgram_endpointing_ms: int = 500
    deepgram_utterance_end_ms: int = 1000

    # ── ElevenLabs — TTS ──────────────────────────────────────────────────────
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "pNInz6obpgDQGcFmaJgB"
    elevenlabs_model_id: str = "eleven_turbo_v2"
    elevenlabs_chunk_size: int = 4096

    # ── Twilio — SMS alerts ───────────────────────────────────────────────────
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""

    # ── Privacy & Data Retention ──────────────────────────────────────────────
    transcript_retention_hours: int = 24
    emotion_retention_days: int = 90
    episodic_memory_retention_days: int = 365
    session_retention_days: int = 30
    session_metadata_retention_days: int = 730

    # ── Safety ────────────────────────────────────────────────────────────────
    # Number of active safety factors required to reach AlertLevel.HIGH.
    # Adults need 3 of 5 factors; minors need only 2 because F4 (vulnerability)
    # is always pre-counted for them, so they effectively start at 1.
    alert_threshold: int = 3  # adult trigger point
    minor_alert_threshold: int = 2  # minor trigger point (was wrongly 3 before)

    # Minimum seconds between guardian SMS alerts for the same user.
    # Prevents alert spam during long sessions with sustained crisis signals.
    # Reserved for future use — not yet implemented in AlertAgent.
    guardian_alert_cooldown_seconds: int = 300

    # ── Rate Limiting ─────────────────────────────────────────────────────────
    rate_limit_ws_messages_per_min: int = 60
    rate_limit_api_calls_per_min: int = 100
    rate_limit_llm_calls_per_hour: int = 10_000
    rate_limit_hash_salt: str = ""

    # ── Background Jobs ───────────────────────────────────────────────────────
    transcript_purge_interval_minutes: int = 60
    reminder_check_interval_seconds: int = 60
    session_cleanup_interval_hours: int = 6

    # ── Emotion Model (wav2vec2) ──────────────────────────────────────────────
    emotion_model_name: str = (
        "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
    )
    emotion_window_ms: int = 500

    # ── Music ─────────────────────────────────────────────────────────────────
    default_volume: int = 50

    # ── ESP32 WebSocket ───────────────────────────────────────────────────────
    ws_ping_interval_seconds: int = 20
    ws_max_audio_frame_bytes: int = 65_536

    # ── Frontend / OAuth ──────────────────────────────────────────────────────
    frontend_base_url: str = "http://localhost:3000"
    cookie_domain: str | None = None
    oauth_state_secret: str = ""

    # ── Google OAuth (Calendar) ───────────────────────────────────────────────
    google_oauth_client_id: str = ""
    google_oauth_client_secret: str = ""
    google_oauth_redirect_uri: str = "http://localhost:8080/api/v1/auth/google/callback"

    # ── Aliases ───────────────────────────────────────────────────────────────

    @property
    def gemini_flash_model(self) -> str:
        """Alias for llm_cheap_model (used by LlmAgent definitions)."""
        return self.llm_cheap_model

    @property
    def gemini_pro_model(self) -> str:
        """Alias for llm_pro_model (used by LlmAgent definitions)."""
        return self.llm_pro_model

    @property
    def cors_origins_list(self) -> list[str]:
        """Return CORS origins as a list, split on commas."""
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment == "production"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset the cached settings singleton. Used in tests."""
    global _settings
    _settings = None
