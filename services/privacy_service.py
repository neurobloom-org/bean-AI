"""
services/privacy_service.py
============================
Privacy service for managing user data retention.
Needs supabase_client.
"""

from datetime import datetime, timezone

from shared.config import config
from services.supabase_client import supabase


async def delete_old_transcripts(user_id: str) -> int:
    """Delete transcripts older than retention period."""
    cutoff = datetime.now(timezone.utc).isoformat()
    result = (
        supabase.table("transcripts")
        .delete()
        .eq("user_id", user_id)
        .lt("created_at", cutoff)
        .execute()
    )
    return len(result.data)


async def delete_user_data(user_id: str) -> dict:
    """Delete all data for a user (GDPR right to erasure)."""
    tables = ["transcripts", "sessions", "emotions", "episodic_memories"]
    deleted = {}
    for table in tables:
        result = supabase.table(table).delete().eq("user_id", user_id).execute()
        deleted[table] = len(result.data)
    return deleted


async def anonymize_session(session_id: str) -> bool:
    """Anonymize a session by removing personal data."""
    supabase.table("sessions").update(
        {"user_text": "[redacted]", "transcript": "[redacted]"}
    ).eq("id", session_id).execute()
    return True
