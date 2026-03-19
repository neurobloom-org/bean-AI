"""
services/safety_service.py
===========================
Safety service for crisis detection and escalation.
Needs twilio_service + supabase_client.
"""

from datetime import datetime, timezone

from shared.enums import CrisisLevel
from services.supabase_client import supabase
from services.twilio_service import send_crisis_alert

CRISIS_L2_KEYWORDS = [
    "kill myself",
    "end my life",
    "suicide",
    "want to die",
    "better off dead",
    "no reason to live",
    "hurt myself",
]

CRISIS_L1_KEYWORDS = [
    "can't cope",
    "falling apart",
    "panic attack",
    "completely overwhelmed",
    "breaking down",
    "hopeless",
    "helpless",
]


def assess_crisis_level(text: str) -> CrisisLevel:
    """Assess the crisis level of a given text."""
    text_lower = text.lower()
    if any(kw in text_lower for kw in CRISIS_L2_KEYWORDS):
        return CrisisLevel.CRITICAL
    if any(kw in text_lower for kw in CRISIS_L1_KEYWORDS):
        return CrisisLevel.ELEVATED
    return CrisisLevel.NONE


async def handle_crisis(user_id: str, text: str, level: CrisisLevel) -> dict:
    """Handle a crisis event — log and escalate if needed."""
    supabase.table("crisis_alerts").insert(
        {
            "user_id": user_id,
            "level": level.value,
            "user_text": text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    ).execute()

    if level == CrisisLevel.CRITICAL:
        guardian = get_guardian_contact(user_id)
        if guardian:
            await send_crisis_alert(guardian, user_id)

    return {"level": level.value, "handled": True}


def get_guardian_contact(user_id: str) -> str | None:
    """Get the guardian contact number for a user."""
    result = (
        supabase.table("users").select("guardian_phone").eq("id", user_id).execute()
    )
    if result.data:
        return result.data[0].get("guardian_phone")
    return None
