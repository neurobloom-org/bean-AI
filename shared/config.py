"""
shared/config.py
----------------
Central configuration for Bean AI system.
Loads all environment variables in one place.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Central config — all environment variables live here."""

    # ── LLM ──────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instant")
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "150"))

    # ── Database ──────────────────────────────────────────
    SUPABASE_URL: str = os.getenv("SUPABASE_URL", "")
    SUPABASE_KEY: str = os.getenv("SUPABASE_KEY", "")

    # ── RAG ───────────────────────────────────────────────
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "faiss_index.bin")
    CHUNKS_PATH: str = os.getenv("CHUNKS_PATH", "chunks.json")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "3"))

    # ── Security ──────────────────────────────────────────
    SECRET_KEY: str = os.getenv("SECRET_KEY", "bean-dev-key")

    # ── Speech ────────────────────────────────────────────
    WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "tiny")

    # ── External Services ─────────────────────────────────
    ELEVENLABS_API_KEY: str = os.getenv("ELEVENLABS_API_KEY", "")
    DEEPGRAM_API_KEY: str = os.getenv("DEEPGRAM_API_KEY", "")
    TWILIO_SID: str = os.getenv("TWILIO_SID", "")
    TWILIO_TOKEN: str = os.getenv("TWILIO_TOKEN", "")


# Single instance imported everywhere
config = Config()

