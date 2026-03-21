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
from google.adk.events import Event, EventActions

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
    """Compute alert level based on active factor count and user age group."""
    threshold = 2 if is_minor else 3

    if active_count == 0:
        return AlertLevel.NONE
    if active_count < threshold:
        return AlertLevel.LOW if active_count == 1 else AlertLevel.MEDIUM
    if active_count == threshold:
        return AlertLevel.HIGH
    return AlertLevel.CRISIS


def _normalize_recent_emotion_labels(recent_emotions: list[object]) -> list[str]:
    """Normalize recent emotion entries into a list of emotion label strings."""
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
        """Read session state, evaluate safety factors, and dispatch alerts."""
        session_id = ctx.session.state.get("session_id", "")
        user_id = ctx.session.state.get("user_id", "")
        transcript = ctx.session.state.get("current_transcript", "")
        current_emotion_str = ctx.session.state.get("current_emotion", "neutral")
        emotion_confidence = float(ctx.session.state.get("emotion_confidence", 0.0))
        is_minor = bool(ctx.session.state.get("is_minor", False))

        raw_dispatched = ctx.session.state.get("alert_dispatched", False)
        already_dispatched = str(raw_dispatched).lower() == "true"

        logger.debug(
            "AlertAgent started [session=%s, user=%s, is_minor=%s, dispatched=%s]",
            session_id,
            user_id,
            is_minor,
            already_dispatched,
        )

        if not transcript or already_dispatched:
            return

        try:
            current_emotion = EmotionLabel(current_emotion_str)
        except ValueError:
            current_emotion = EmotionLabel.NEUTRAL

        active_factors: list[str] = []

        has_crisis, _ = check_crisis_keywords(transcript, is_minor)
        if has_crisis:
            active_factors.append(AlertFactor.F1_CRISIS_KEYWORD.value)

        if current_emotion in EmotionLabel.negative() and emotion_confidence > 0.5:
            active_factors.append(AlertFactor.F2_NEGATIVE_EMOTION.value)

        recent_emotions_raw = ctx.session.state.get("recent_emotions", "[]")

        try:
            if isinstance(recent_emotions_raw, str):
                recent_emotions: list[object] = json.loads(recent_emotions_raw)
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

        if is_minor:
            active_factors.append(AlertFactor.F4_VULNERABILITY.value)

        has_explicit, _ = check_explicit_statement(transcript)
        if has_explicit:
            active_factors.append(AlertFactor.F5_EXPLICIT_STATEMENT.value)

        active_factors = list(dict.fromkeys(active_factors))

        active_count = len(active_factors)
        alert_level = _compute_alert_level(active_count, is_minor)

        ctx.session.state["alert_level"] = alert_level.value
        ctx.session.state["alert_active_count"] = str(active_count)

        logger.info(
            "Alert factors evaluated [session=%s, user=%s, level=%s, count=%s, "
            "factors=%s]",
            session_id,
            user_id,
            alert_level.value,
            active_count,
            active_factors,
        )

        if alert_level in (AlertLevel.HIGH, AlertLevel.CRISIS) and not already_dispatched:
            await self._dispatch_alert(
                ctx=ctx,
                active_factors=active_factors,
                alert_level=alert_level,
            )

        yield Event(
            author=self.name,
            actions=EventActions(
                state_delta={
                    "alert_level": alert_level.value,
                    "alert_active_count": str(active_count),
                }
            ),
        )

    async def _dispatch_alert(
        self,
        ctx: InvocationContext,
        active_factors: list[str],
        alert_level: AlertLevel,
    ) -> None:
        """Dispatch guardian alert and persist the alert row."""
        session_id = ctx.session.state.get("session_id", "")
        user_id = ctx.session.state.get("user_id", "")

        try:
            client = await get_service_client()

            guardian_result = (
                await client.table("guardian_links")
                .select("guardian_user_id")
                .eq("patient_user_id", user_id)
                .eq("can_view_alerts", True)
                .execute()
            )

            profile_result = (
                await client.table("user_profiles")
                .select("display_name")
                .eq("user_id", user_id)
                .maybe_single()
                .execute()
            )

            profile_data = profile_result.data if profile_result is not None else None
            display_name: str = (
                str(profile_data.get("display_name", "your patient"))
                if isinstance(profile_data, dict)
                else "your patient"
            )

            sms_sent = False
            guardian_notified_at = None

            guardian_data = guardian_result.data if guardian_result is not None else []

            for link in guardian_data or []:
                if not isinstance(link, dict):
                    continue

                guardian_id = link.get("guardian_user_id")
                if not guardian_id:
                    continue

                try:
                    guardian_auth = await client.auth.admin.get_user_by_id(
                        str(guardian_id)
                    )
                    user_metadata = (
                        guardian_auth.user.user_metadata
                        if guardian_auth.user is not None
                        else None
                    )
                    phone: str | None = (
                        str(user_metadata.get("phone"))
                        if isinstance(user_metadata, dict)
                        and user_metadata.get("phone") is not None
                        else None
                    )

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

            ctx.session.state["post_alert_message"] = get_post_alert_message()
            ctx.session.state["alert_dispatched"] = "true"

            logger.warning(
                "ALERT DISPATCHED [session=%s, user=%s, level=%s, factors=%s, "
                "sms_sent=%s]",
                session_id,
                user_id,
                alert_level.value,
                active_factors,
                sms_sent,
            )

        except Exception as exc:
            logger.error(
                "Alert dispatch failed [session=%s, user=%s]: %s",
                session_id,
                user_id,
                exc,
            )

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

            ctx.session.state["post_alert_message"] = get_post_alert_message()
            ctx.session.state["alert_dispatched"] = "true"


alert_agent = AlertAgent(name="alert_agent")