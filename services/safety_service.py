"""BEAN AI v1 — Safety / Alert service.

Uses Gemini Pro (never Flash) for all safety decisions.

Privacy:
  - Trigger phrases and crisis keywords are NEVER stored in the database
  - Only alert_level and alert_factors (factor codes) are persisted
  - SMS to guardian contains only a generic concern notice — no conversation content
"""

import asyncio
import logging
import random
from typing import Any

from services.llm_service import assess_safety
from services.supabase_client import get_service_client
from shared.enums import AlertFactor, AlertLevel, EmotionLabel
from shared.exceptions import CrisisDetectedError


logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Crisis keywords — fast pre-screen before any LLM call
# ─────────────────────────────────────────────────────────────────────────────
CRISIS_KEYWORDS: frozenset[str] = frozenset(
    {
        "kill myself",
        "end my life",
        "want to die",
        "suicide",
        "hurt myself",
        "self harm",
        "cut myself",
        "don't want to be here",
        "no reason to live",
        "better off dead",
        "overdose",
        "want to hurt",
        "going to hurt",
        "plan to hurt",
    }
)

EXPLICIT_STATEMENT_KEYWORDS: frozenset[str] = frozenset(
    {
        "i will kill",
        "i am going to kill",
        "i'm going to end",
        "i decided to",
        "tonight i",
        "i have a plan",
        "i already took",
        "i already cut",
        "i've been cutting",
    }
)

_POST_ALERT_MESSAGES = [
    "I want you to know I really care about you, and I've let someone know so they can support you too. 💙",
    "You're not alone in this. Someone who cares about you has been notified. I'm right here with you.",
    "I've let a trusted person know because I care about your safety. Please know you matter so much.",
]

SAFETY_SYSTEM = """You are a clinical safety analyst for a mental health companion robot.
Your role is to assess whether a conversation turn indicates potential risk for a
vulnerable young person (teenager or young adult with mental health conditions).

Be sensitive — false negatives are dangerous. When in doubt, escalate.

Assess these five factors:
  F1 - Crisis keyword: explicit self-harm or suicidal language
  F2 - Negative emotion: sustained negative emotional state
  F3 - Escalation pattern: worsening mood across multiple turns
  F4 - Vulnerability: user has known risk factors
  F5 - Explicit statement: direct statement of intent or hopelessness

Respond ONLY with this exact JSON — no markdown, no preamble:
{
  "alert_level": "none|low|medium|high|crisis",
  "factors": ["f1_crisis_keyword", "f2_negative_emotion", ...],
  "requires_immediate_action": true/false,
  "suggested_response_type": "supportive|redirect|crisis_resources|alert_guardian"
}"""

SAFETY_PROMPT = """Conversation context (do not include in response):
User said: {user_text}
Detected emotion: {emotion}
Recent emotion trend: {emotion_trend}
Turn number in session: {turn_number}

Assess the safety level for this interaction."""


# ─────────────────────────────────────────────────────────────────────────────
# Standalone helper functions (used by agents)
# ─────────────────────────────────────────────────────────────────────────────


def check_crisis_keywords(text: str, is_minor: bool = False) -> tuple[bool, list[str]]:
    """Fast keyword pre-screen. Returns (detected: bool, matched_keywords: list)."""
    text_lower = text.lower()
    matched = [kw for kw in CRISIS_KEYWORDS if kw in text_lower]
    return bool(matched), matched


def check_explicit_statement(text: str) -> tuple[bool, list[str]]:
    """Check for explicit statements of intent. Returns (detected, matched_keywords)."""
    text_lower = text.lower()
    matched = [kw for kw in EXPLICIT_STATEMENT_KEYWORDS if kw in text_lower]
    return bool(matched), matched


def get_post_alert_message() -> str:
    """Return a randomised post-alert message for transparency."""
    return random.choice(_POST_ALERT_MESSAGES)


# ─────────────────────────────────────────────────────────────────────────────
# SafetyService class
# ─────────────────────────────────────────────────────────────────────────────


class SafetyService:
    """Safety assessment and alert escalation.

    Guardian SMS dispatch is owned exclusively by AlertAgent._dispatch_alert().
    This service is responsible for:
      - LLM-based safety assessment (Gemini Pro)
      - Persisting alert records to the DB
      - Raising CrisisDetectedError for CRISIS-level events
    """

    async def assess_turn(
        self,
        user_id: str,
        session_id: str,
        user_text: str,
        emotion: str,
        emotion_trend: list[str],
        turn_number: int,
        vulnerability_flag: bool = False,
    ) -> dict[str, Any]:
        """Assess a conversation turn for safety concerns.

        Returns the assessment dict. Automatically creates DB record.

        Privacy: user_text is sent to Gemini Pro but NOT stored.
        Only the resulting alert_level and factors are persisted.
        """
        factors: list[str] = []

        # F4 — pre-set vulnerability flag
        if vulnerability_flag:
            factors.append(AlertFactor.F4_VULNERABILITY.value)

        # Fast pre-screen for crisis keywords
        has_crisis_keyword, _ = check_crisis_keywords(user_text)
        if has_crisis_keyword:
            factors.append(AlertFactor.F1_CRISIS_KEYWORD.value)

        # F2 — negative emotion
        try:
            emotion_enum = EmotionLabel(emotion)
            if emotion_enum in EmotionLabel.negative():
                factors.append(AlertFactor.F2_NEGATIVE_EMOTION.value)
        except ValueError:
            pass

        # F3 — escalation pattern (3+ consecutive negative emotions)
        negative_labels = {e.value for e in EmotionLabel.negative()}
        if len(emotion_trend) >= 3 and all(
            e in negative_labels for e in emotion_trend[-3:]
        ):
            factors.append(AlertFactor.F3_ESCALATION.value)

        # F5 — explicit statement
        has_explicit, _ = check_explicit_statement(user_text)
        if has_explicit:
            factors.append(AlertFactor.F5_EXPLICIT_STATEMENT.value)

        # Use Pro LLM for detailed assessment
        prompt = SAFETY_PROMPT.format(
            user_text=user_text[:400],
            emotion=emotion,
            emotion_trend=", ".join(emotion_trend[-5:]),
            turn_number=turn_number,
        )

        try:
            assessment = await asyncio.wait_for(
                assess_safety(prompt=prompt, system=SAFETY_SYSTEM),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            logger.error(
                "Safety LLM assessment timed out after 15s for user %s", user_id
            )
            assessment = {"alert_level": "none", "factors": factors}
        except Exception as exc:
            logger.error("Safety LLM assessment failed: %s", exc)
            if has_crisis_keyword:
                assessment = {
                    "alert_level": "high",
                    "factors": factors,
                    "requires_immediate_action": True,
                    "suggested_response_type": "crisis_resources",
                }
            else:
                return {"alert_level": "none", "factors": factors}

        # Merge LLM-detected factors
        for f in assessment.get("factors", []):
            if f not in factors:
                factors.append(f)

        alert_level_str = assessment.get("alert_level", "none")
        try:
            alert_level = AlertLevel(alert_level_str)
        except ValueError:
            alert_level = AlertLevel.NONE

        # Persist alert if not "none"
        if alert_level != AlertLevel.NONE:
            await self._create_alert(
                user_id=user_id,
                session_id=session_id,
                alert_level=alert_level,
                factors=factors,
            )
            await self._notify_guardian(user_id=user_id, alert_level=alert_level)

        # Raise immediately for crisis
        if alert_level == AlertLevel.CRISIS:
            raise CrisisDetectedError(
                f"Crisis detected for user {user_id}: factors={factors}"
            )

        assessment["factors"] = factors
        assessment["alert_level"] = alert_level.value
        return assessment

    async def _create_alert(
        self,
        user_id: str,
        session_id: str,
        alert_level: AlertLevel,
        factors: list[str],
    ) -> str | None:
        """Persist alert record only. Guardian SMS dispatch is owned by AlertAgent."""
        client = await get_service_client()
        try:
            result = (
                await client.table("alerts")
                .insert(
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "alert_level": alert_level.value,
                        "alert_factors": factors,
                    }
                )
                .execute()
            )
            alert_id = result.data[0]["id"] if result.data else None
            logger.warning(
                "Alert created [user=%s level=%s factors=%s]",
                user_id,
                alert_level.value,
                factors,
            )
            return alert_id

        except Exception as exc:
            logger.error("Failed to create alert record: %s", exc)
            return None
    async def _notify_guardian(self, user_id: str, alert_level: AlertLevel) -> None:
        """Send privacy-safe SMS to the user's guardian/doctor."""
        try:
            client = await get_service_client()

            guardian_result = (
                await client.table("guardian_links")
                .select("guardian_user_id, patient_user_id")
                .eq("patient_user_id", user_id)
                .eq("can_view_alerts", True)
                .execute()
            )

            if not guardian_result.data:
                logger.warning("No guardian found for user %s", user_id)
                return

            for link in guardian_result.data:
                guardian_id = link["guardian_user_id"]
                try:
                    guardian_auth = await asyncio.wait_for(
                        get_service_client(),
                        timeout=5.0,
                    )
                    user_resp = (
                        await guardian_auth.auth.admin.get_user_by_id(guardian_id)
                    )
                    phone = (user_resp.user.user_metadata or {}).get("phone")
                    if phone:
                        from services.twilio_service import send_guardian_sms
                        if alert_level == AlertLevel.CRISIS:
                            body = "URGENT: BEAN AI has detected a crisis situation. Please check on your patient immediately."
                        else:
                            body = "BEAN AI has detected a concern with your patient. Please check in with them when you can."
                        await send_guardian_sms(to=phone, body=body)
                        logger.warning(
                            "SMS sent to guardian %s for user %s [level=%s]",
                            guardian_id, user_id, alert_level.value,
                        )
                    else:
                        logger.warning("AlertAgent: guardian phone missing [guardian=%s]", guardian_id)
                except Exception as exc:
                    logger.error("AlertAgent: guardian SMS failed [guardian=%s]: %s", guardian_id, exc)

        except Exception as exc:
            logger.error("_notify_guardian failed for user %s: %s", user_id, exc)

# ── Singleton ─────────────────────────────────────────────────────────────────
safety_service = SafetyService()
