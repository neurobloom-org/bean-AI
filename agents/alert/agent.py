"""BEAN AI — Alert Agent

Custom BaseAgent for real-time safety monitoring.

What this agent does:
- Runs parallel safety monitoring using session.state data
- Evaluates a 5-factor safety scoring system
- Uses a 3-of-5 threshold for adults
- Uses a 2-of-5 threshold for minors because F4 (vulnerability) is auto-counted
- Dispatches guardian SMS when alert threshold is met
- Persists alert records to the database
- Prevents duplicate dispatches using session.state["alert_dispatched"]

Factors:
- F1: crisis_keyword
- F2: negative_emotion
- F3: escalation_pattern
- F4: vulnerability
- F5: explicit_statement
"""

import json
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from services.safety_service import (
    check_crisis_keywords,
    check_explicit_statement,
    get_post_alert_message,
)
from services.supabase_client import get_service_client
from services.twilio_service import send_guardian_alert
from shared.enums import AlertFactor, AlertLevel, EmotionLabel

logger = logging.getLogger(__name__)


def _compute_alert_level(active_count: int, is_minor: bool) -> AlertLevel:
    """Compute alert level based on active factor count and user age group.

    Threshold rules:
    - Adults: threshold = 3
    - Minors: threshold = 2 (because vulnerability is auto-counted)

    Levels:
    - 0 => NONE
    - below threshold => LOW or MEDIUM
    - == threshold => HIGH
    - > threshold => CRISIS
    """
    threshold = 2 if is_minor else 3

    if active_count == 0:
        return AlertLevel.NONE

    if active_count < threshold:
        return AlertLevel.LOW if active_count == 1 else AlertLevel.MEDIUM

    if active_count == threshold:
        return AlertLevel.HIGH

    return AlertLevel.CRISIS


def _normalize_recent_emotion_labels(recent_emotions: list[object]) -> list[str]:
    """Normalize recent emotion entries into a simple list of emotion label strings.

    This supports multiple possible input shapes, for example:
    - ["sad", "fear", "anger"]
    - [["sad", 0.90], ["fear", 0.84], ["anger", 0.91]]
    - [{"label": "sad"}, {"label": "fear"}, {"label": "anger"}]

    Returns a normalized list such as:
    - ["sad", "fear", "anger"]
    """
    normalized: list[str] = []

    for item in recent_emotions:
        if isinstance(item, str):
            normalized.append(item)

        elif isinstance(item, (list, tuple)) and item:
            normalized.append(str(item[0]))

        elif isinstance(item, dict):
            label = item.get("label") or item.get("emotion") or item.get("name")
            normalized.append(str(label) if label is not None else "")

        else:
            normalized.append("")

    return normalized


class AlertAgent(BaseAgent):
    """Custom BaseAgent for real-time safety monitoring and guardian alert dispatch."""

    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Main async execution logic for the alert agent.

        Reads data from session.state, evaluates all safety factors,
        updates session alert state, and dispatches alerts if needed.
        """
        session_id = ctx.session.state.get("session_id", "")
        user_id = ctx.session.state.get("user_id", "")
        transcript = ctx.session.state.get("current_transcript", "")
        current_emotion_str = ctx.session.state.get("current_emotion", "neutral")
        emotion_confidence = float(ctx.session.state.get("emotion_confidence", 0.0))
        is_minor = bool(ctx.session.state.get("is_minor", False))

        # Make the dispatched flag tolerant to True / "true" / "True"
        raw_dispatched = ctx.session.state.get("alert_dispatched", False)
        already_dispatched = str(raw_dispatched).lower() == "true"

        logger.debug(
            "AlertAgent started [session=%s, user=%s, is_minor=%s, dispatched=%s]",
            session_id,
            user_id,
            is_minor,
            already_dispatched,
        )

        # If there is no transcript, there is nothing to evaluate.
        # If alert already dispatched, do not dispatch again.
        if not transcript or already_dispatched:
            return

        # Parse current emotion safely. Default to NEUTRAL on bad input.
        try:
            current_emotion = EmotionLabel(current_emotion_str)
        except ValueError:
            current_emotion = EmotionLabel.NEUTRAL

        active_factors: list[str] = []

        # ─────────────────────────────────────────────────────────────
        # F1 — Crisis keyword
        # Triggered when safety service detects crisis language
        # ─────────────────────────────────────────────────────────────
        has_crisis, _ = check_crisis_keywords(transcript, is_minor)
        if has_crisis:
            active_factors.append(AlertFactor.F1_CRISIS_KEYWORD.value)

        # ─────────────────────────────────────────────────────────────
        # F2 — Negative emotion
        # Triggered when current emotion is negative with confidence > 0.5
        # ─────────────────────────────────────────────────────────────
        if current_emotion in EmotionLabel.negative() and emotion_confidence > 0.5:
            active_factors.append(AlertFactor.F2_NEGATIVE_EMOTION.value)

        # ─────────────────────────────────────────────────────────────
        # F3 — Escalation pattern
        # Triggered when the last 3 recent emotions are all negative
        # ─────────────────────────────────────────────────────────────
        recent_emotions_raw = ctx.session.state.get("recent_emotions", "[]")

        try:
            if isinstance(recent_emotions_raw, str):
                recent_emotions = json.loads(recent_emotions_raw)
            elif isinstance(recent_emotions_raw, list):
                recent_emotions = recent_emotions_raw
            else:
                recent_emotions = []
        except (json.JSONDecodeError, ValueError, TypeError):
            recent_emotions = []

        normalized_recent = _normalize_recent_emotion_labels(recent_emotions)
        last_three = normalized_recent[-3:]
        negative_labels = {emotion.value for emotion in EmotionLabel.negative()}

        if len(last_three) == 3 and all(label in negative_labels for label in last_three):
            active_factors.append(AlertFactor.F3_ESCALATION.value)

        # ─────────────────────────────────────────────────────────────
        # F4 — Vulnerability
        # Auto-counted for minors
        # ─────────────────────────────────────────────────────────────
        if is_minor:
            active_factors.append(AlertFactor.F4_VULNERABILITY.value)

        # ─────────────────────────────────────────────────────────────
        # F5 — Explicit statement
        # Triggered when explicit self-harm / danger statement is detected
        # ─────────────────────────────────────────────────────────────
        has_explicit, _ = check_explicit_statement(transcript)
        if has_explicit:
            active_factors.append(AlertFactor.F5_EXPLICIT_STATEMENT.value)

        # Remove duplicates while preserving order.
        active_factors = list(dict.fromkeys(active_factors))

        active_count = len(active_factors)
        alert_level = _compute_alert_level(active_count, is_minor)

        # Store computed state for downstream agents / orchestration.
        ctx.session.state["alert_level"] = alert_level.value
        ctx.session.state["alert_active_count"] = str(active_count)

        logger.info(
            "Alert factors evaluated [session=%s, user=%s, level=%s, count=%s, factors=%s]",
            session_id,
            user_id,
            alert_level.value,
            active_count,
            active_factors,
        )

        # Dispatch only when threshold is met and not already sent.
        if alert_level in (AlertLevel.HIGH, AlertLevel.CRISIS) and not already_dispatched:
            await self._dispatch_alert(
                ctx=ctx,
                active_factors=active_factors,
                alert_level=alert_level,
                transcript=transcript,
            )

        # Emit state_delta so the rest of the pipeline can observe updates.
        yield Event(
            author=self.name,
            actions={
                "state_delta": {
                    "alert_level": alert_level.value,
                    "alert_active_count": str(active_count),
                }
            },
        )

    async def _dispatch_alert(
        self,
        ctx: InvocationContext,
        active_factors: list[str],
        alert_level: AlertLevel,
        transcript: str,
    ) -> None:
        """Dispatch guardian alert and persist the alert row.

        Success path:
        - tries to find guardians who can view alerts
        - tries to send SMS
        - inserts alert row
        - marks alert_dispatched=true
        - sets post_alert_message

        Failure path:
        - still tries to insert alert row with sms_sent=False
        - still marks alert_dispatched=true to prevent duplicate dispatch loops
        """
        session_id = ctx.session.state.get("session_id", "")
        user_id = ctx.session.state.get("user_id", "")

        try:
            client = await get_service_client()

            # Find guardians linked to this user who are allowed to view alerts.
            guardian_result = (
                await client.table("guardian_links")
                .select("guardian_user_id")
                .eq("patient_user_id", user_id)
                .eq("can_view_alerts", True)
                .execute()
            )

            # Fetch patient display name for SMS message.
            profile_result = (
                await client.table("user_profiles")
                .select("display_name")
                .eq("user_id", user_id)
                .maybe_single()
                .execute()
            )
            display_name = (profile_result.data or {}).get("display_name", "your patient")

            sms_sent = False
            guardian_notified_at = None

            # Try sending SMS to all linked guardians with valid phone numbers.
            for link in guardian_result.data or []:
                guardian_id = link.get("guardian_user_id")
                if not guardian_id:
                    continue

                try:
                    guardian_auth = await client.auth.admin.get_user_by_id(guardian_id)
                    phone = (guardian_auth.user.user_metadata or {}).get("phone")

                    if not phone:
                        logger.warning(
                            "Guardian phone missing [guardian=%s]",
                            str(guardian_id)[:8],
                        )
                        continue

                    await send_guardian_alert(
                        guardian_phone=phone,
                        user_display_name=display_name,
                        alert_level=alert_level.value,
                        session_id=session_id,
                    )

                    sms_sent = True
                    guardian_notified_at = datetime.now(UTC).isoformat()

                except Exception as exc:
                    logger.error(
                        "Guardian SMS failed [guardian=%s]: %s",
                        str(guardian_id)[:8],
                        exc,
                    )

            # Persist alert row regardless of whether SMS succeeded.
            await (
                client.table("alerts")
                .insert(
                    {
                        "session_id": session_id,
                        "user_id": user_id,
                        "alert_level": alert_level.value,
                        "alert_factors": active_factors,
                        "notified_guardian": sms_sent,
                        "guardian_notified_at": guardian_notified_at,
                        "sms_sent": sms_sent,
                    }
                )
                .execute()
            )

            # Mark session so duplicate SMS / duplicate alert rows are prevented.
            ctx.session.state["post_alert_message"] = get_post_alert_message()
            ctx.session.state["alert_dispatched"] = "true"

            logger.warning(
                "ALERT DISPATCHED [session=%s, user=%s, level=%s, factors=%s, sms_sent=%s]",
                session_id,
                user_id,
                alert_level.value,
                active_factors,
                sms_sent,
            )

        except Exception as exc:
            logger.error("Alert dispatch failed [session=%s, user=%s]: %s", session_id, user_id, exc)

            # Best-effort fallback:
            # still try to log the alert in DB even if SMS path failed.
            try:
                client = await get_service_client()

                await (
                    client.table("alerts")
                    .insert(
                        {
                            "session_id": session_id,
                            "user_id": user_id,
                            "alert_level": alert_level.value,
                            "alert_factors": active_factors,
                            "notified_guardian": False,
                            "guardian_notified_at": None,
                            "sms_sent": False,
                        }
                    )
                    .execute()
                )

            except Exception as db_exc:
                logger.error(
                    "Failed to persist fallback alert record [session=%s, user=%s]: %s",
                    session_id,
                    user_id,
                    db_exc,
                )

            # Even in failure path, mark dispatched to avoid duplicate repeats.
            # This matches the lead's warning about respecting the alert_dispatched flag.
            ctx.session.state["post_alert_message"] = get_post_alert_message()
            ctx.session.state["alert_dispatched"] = "true"


# Singleton instance used by the app/orchestrator.
alert_agent = AlertAgent(name="alert_agent")