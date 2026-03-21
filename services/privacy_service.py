"""BEAN AI — Privacy enforcement service.

Core principle: BEAN is a mental health device for minors and young adults.
Data minimisation and user control are non-negotiable.

What NEVER gets stored:
  ✗  Raw audio bytes / voice recordings
  ✗  Word-for-word transcripts (beyond 24h active window)
  ✗  Raw emotion probability scores / logits
  ✗  Personally identifiable conversation content long-term
  ✗  TTS audio (streamed to ESP32, never written to disk)

What gets stored (minimised / abstracted):
  ✓  Extracted facts only      → "likes football", "has a test tomorrow"
  ✓  Emotion label only        → "sad" (not audio, not raw probabilities)
  ✓  Alert level + factors     → timestamp + level (not trigger phrase)
  ✓  Session metadata          → timestamps, duration, turn count (not content)
  ✓  Episodic vector embedding → cosine-searchable (source text discarded)

Guardian / doctor access:
  ✓  Emotion trend graphs (aggregated, anonymised)
  ✓  Alert summary (level + timestamp, not conversation)
  ✗  Raw transcripts (never — enforced at DB level via RLS)
  ✗  Conversation content
"""

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from services.supabase_client import get_service_client
from shared.config import get_settings

logger = logging.getLogger(__name__)

# Fields that should NEVER be sent to the database
_BANNED_FIELDS = frozenset(
    {
        "raw_audio",
        "audio_bytes",
        "audio_b64",
        "audio_data",
        "raw_transcript",
        "full_transcript",
        "conversation_text",
        "voice_data",
    }
)


class PrivacyService:
    """Enforces BEAN AI's privacy-first data policy."""

    # ─────────────────────────────────────────────────────────────────────────
    # Transcript lifecycle (24h temporary storage)
    # ─────────────────────────────────────────────────────────────────────────

    async def store_transcript_turn(
        self,
        session_id: str,
        user_id: str,
        speaker: str,  # "user" | "assistant"
        text: str,
        max_length: int = 2000,
    ) -> str:
        """Store a transcript turn that auto-expires after 24h.

        Text is truncated to max_length to limit PII footprint.
        Returns the inserted record ID.
        """
        settings = get_settings()
        expires_at = datetime.now(UTC) + timedelta(
            hours=settings.transcript_retention_hours
        )

        # Truncate — limit PII exposure in temporary store
        stored_text = text[:max_length] if len(text) > max_length else text

        client = await get_service_client()
        result = (
            await client.table("session_transcripts")
            .insert(
                {
                    "session_id": session_id,
                    "user_id": user_id,
                    "speaker": speaker,
                    "text": stored_text,
                    "expires_at": expires_at.isoformat(),
                }
            )
            .execute()
        )
        return result.data[0]["id"] if result.data else ""

    async def get_recent_transcript(
        self,
        session_id: str,
        max_turns: int = 10,
    ) -> list[dict[str, str]]:
        """Retrieve recent transcript turns for in-session context.

        Only returns turns that haven't expired yet.
        Returns list of {"speaker": ..., "text": ...} dicts.
        """
        now = datetime.now(UTC).isoformat()
        client = await get_service_client()
        result = (
            await client.table("session_transcripts")
            .select("speaker, text, created_at")
            .eq("session_id", session_id)
            .gt("expires_at", now)
            .order("created_at", desc=False)
            .limit(max_turns)
            .execute()
        )
        return [
            {"speaker": r["speaker"], "text": r["text"]} for r in (result.data or [])
        ]

    async def purge_expired_transcripts(self) -> int:
        """Delete all transcripts past their expiry. Called by background job.

        Returns count of deleted rows.
        """
        now = datetime.now(UTC).isoformat()
        client = await get_service_client()
        result = (
            await client.table("session_transcripts")
            .delete()
            .lt("expires_at", now)
            .execute()
        )
        count = len(result.data) if result.data else 0
        if count:
            logger.info("Privacy purge: deleted %d expired transcript turns", count)
        return count

    # ─────────────────────────────────────────────────────────────────────────
    # Right to be forgotten — GDPR cascade delete
    # ─────────────────────────────────────────────────────────────────────────

    async def delete_all_user_data(self, user_id: str) -> dict[str, int]:
        """Full right-to-be-forgotten: hard-delete all data for a user.

        Order matters — delete child tables before parents to avoid FK errors.
        Returns per-table deletion counts for audit log.
        """
        client = await get_service_client()
        deleted: dict[str, int] = {}

        # Delete in dependency order
        ordered_tables = [
            "session_transcripts",
            "episodic_memories",
            "emotion_events",
            "alerts",
            "tasks",
            "sessions",
            "user_profiles",
            "guardian_links",  # remove as both guardian and patient
        ]

        for table in ordered_tables:
            try:
                result = (
                    await client.table(table).delete().eq("user_id", user_id).execute()
                )
                deleted[table] = len(result.data) if result.data else 0
                logger.info(
                    "GDPR delete: table=%s user=%s rows=%d",
                    table,
                    user_id,
                    deleted[table],
                )
            except Exception as exc:
                logger.error("GDPR delete failed [table=%s]: %s", table, exc)
                deleted[table] = -1  # mark as failed

        # Remove guardian_links where this user is the guardian
        try:
            await (
                client.table("guardian_links")
                .delete()
                .eq("guardian_user_id", user_id)
                .execute()
            )
        except Exception as exc:
            logger.error("GDPR delete guardian links failed: %s", exc)

        # Delete from Supabase Auth (this is the final step)
        try:
            await client.auth.admin.delete_user(user_id)
            logger.info("GDPR delete: auth user %s removed", user_id)
        except Exception as exc:
            logger.error("GDPR delete: auth deletion failed for %s: %s", user_id, exc)

        return deleted

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers for safe LLM prompt building
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def sanitize_for_llm(data: dict[str, Any]) -> dict[str, Any]:
        """Strip banned fields before including data in an LLM prompt."""
        return {k: v for k, v in data.items() if k not in _BANNED_FIELDS}

    @staticmethod
    def truncate_for_prompt(text: str, max_chars: int = 500) -> str:
        """Truncate text to reduce PII exposure in LLM API calls."""
        if not text:
            return ""
        return text[:max_chars] if len(text) > max_chars else text

    @staticmethod
    def redact_sensitive_patterns(text: str) -> str:
        """Basic redaction of obvious PII patterns before LLM submission.

        Covers: phone numbers, email addresses, ID numbers.
        Not a replacement for proper PII detection — used as a best-effort layer.
        """
        import re

        # Phone numbers
        text = re.sub(r"\b\+?[\d\s\-().]{10,15}\b", "[PHONE]", text)
        # Email addresses
        text = re.sub(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b", "[EMAIL]", text)
        # National ID / passport patterns (Sri Lanka NIC as example)
        text = re.sub(r"\b\d{9}[VXvx]\b", "[NIC]", text)
        text = re.sub(r"\b\d{12}\b", "[ID]", text)

        return text

    # ─────────────────────────────────────────────────────────────────────────
    # Data retention checks
    # ─────────────────────────────────────────────────────────────────────────

    async def purge_expired_emotion_events(self) -> int:
        """Delete emotion events older than retention period."""
        now = datetime.now(UTC).isoformat()
        client = await get_service_client()
        result = (
            await client.table("emotion_events")
            .delete()
            .lt("expires_at", now)
            .execute()
        )
        count = len(result.data) if result.data else 0
        if count:
            logger.info("Privacy purge: deleted %d expired emotion events", count)
        return count

    async def purge_expired_episodic_memories(self) -> int:
        """Delete episodic memory vectors past their expiry (if set)."""
        now = datetime.now(UTC).isoformat()
        client = await get_service_client()
        result = (
            await client.table("episodic_memories")
            .delete()
            .lt("expires_at", now)
            .not_.is_("expires_at", "null")
            .execute()
        )
        count = len(result.data) if result.data else 0
        if count:
            logger.info("Privacy purge: deleted %d expired episodic memories", count)
        return count


# ── Singleton ─────────────────────────────────────────────────────────────────
privacy_service = PrivacyService()
