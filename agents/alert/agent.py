"""BEAN AI — Alert Agent

Custom BaseAgent for real-time safety monitoring.

What this agent does:
- Evaluates a 5-factor safety scoring system on every conversation turn
- Compares the active factor count against configurable per-age-group thresholds
- Dispatches guardian SMS when the alert threshold is met
- Persists alert records to the database (both on success and failure)
- Prevents duplicate dispatches within a session using session.state["alert_dispatched"]

Factors:
    F1: crisis_keyword      — explicit self-harm / suicidal language
    F2: negative_emotion    — sustained negative emotional state (confidence > 0.5)
    F3: escalation_pattern  — 3 consecutive negative emotion labels
    F4: vulnerability       — user is a minor (auto-counted, always active for minors)
    F5: explicit_statement  — direct statement of intent or hopelessness

Thresholds (from settings, not hardcoded):
    alert_threshold        — adults (default 3)
    minor_alert_threshold  — minors (default 2); lower because F4 is always pre-counted

AlertLevel mapping (example, adult threshold=3):
    0 factors  → NONE
    1 factor   → LOW
    2 factors  → MEDIUM
    3 factors  → HIGH      ← guardian SMS dispatched
    4+ factors → CRISIS    ← guardian SMS dispatched

For minors (threshold=2) there is no MEDIUM level — 2 factors goes straight to HIGH.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any

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
from shared.config import get_settings
from shared.enums import AlertFactor, AlertLevel, EmotionLabel

logger = logging.getLogger(__name__)

# ── Tuning constants ──────────────────────────────────────────────────────────

# Timeout for each individual Supabase call inside _dispatch_alert.
# Guardian SMS is safety-critical but we cannot block indefinitely.
_DB_TIMEOUT_SECONDS: float = 8.0

# Timeout for the Supabase auth.admin.get_user_by_id call (guardian phone lookup).
_AUTH_LOOKUP_TIMEOUT_SECONDS: float = 5.0


# ── Pure helpers ──────────────────────────────────────────────────────────────


def _compute_alert_level(
    active_count: int,
    *,
    threshold: int,
) -> AlertLevel:
    """Compute AlertLevel from the number of active factors and the threshold.

    Args:
        active_count: Number of distinct safety factors that fired this turn.
        threshold:    Minimum count to reach AlertLevel.HIGH.
                      Comes from settings: alert_threshold (adults) or
                      minor_alert_threshold (minors).

    Returns:
        AlertLevel enum value.

    Level mapping:
        0                    → NONE
        1                    → LOW
        2 to threshold-1     → MEDIUM  (only reachable when threshold >= 3)
        threshold            → HIGH    ← guardian SMS threshold
        threshold+1 and above → CRISIS
    """
    if active_count <= 0:
        return AlertLevel.NONE
    if active_count < threshold:
        return AlertLevel.LOW if active_count == 1 else AlertLevel.MEDIUM
    if active_count == threshold:
        return AlertLevel.HIGH
    return AlertLevel.CRISIS


def _parse_emotion_confidence(raw: Any) -> float:
    """Safely parse emotion_confidence from session state.

    Returns 0.0 on any parse failure rather than raising.
    Booleans are rejected — True/False as a confidence value is a data error.
    """
    if isinstance(raw, bool):
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    # Clamp to [0.0, 1.0] — values outside this range are nonsensical.
    return max(0.0, min(1.0, value))


def _parse_already_dispatched(raw: Any) -> bool:
    """Safely parse the alert_dispatched flag from session state.

    Handles the three forms it may arrive in:
      - bool True/False (stored directly as Python bool)
      - str "true"/"false"/"1"/"0" (stored via state_delta)
      - anything else → False (safe default)
    """
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "1", "yes"}
    return False


def _normalize_recent_emotion_labels(recent_emotions: list[object]) -> list[str]:
    """Normalize recent emotion entries into a flat list of label strings.

    Handles three formats the orchestrator may store in session state:
      - str:              "sad"
      - list/tuple:       ["sad", 0.9]  (label + confidence pair)
      - dict:             {"label": "sad", "confidence": 0.9}
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


# ── Agent ─────────────────────────────────────────────────────────────────────


class AlertAgent(BaseAgent):
    """Custom BaseAgent for real-time safety monitoring and guardian alert dispatch."""

    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Read session state, evaluate safety factors, and dispatch alerts."""
        state = ctx.session.state

        session_id: str = str(state.get("session_id") or "")
        user_id: str = str(state.get("user_id") or "")
        transcript: str = str(state.get("current_transcript") or "")
        current_emotion_str: str = str(state.get("current_emotion") or "neutral")
        emotion_confidence: float = _parse_emotion_confidence(
            state.get("emotion_confidence", 0.0)
        )
        is_minor: bool = bool(state.get("is_minor", False))
        already_dispatched: bool = _parse_already_dispatched(
            state.get("alert_dispatched", False)
        )

        logger.debug(
            "AlertAgent started [session=%s user=%s is_minor=%s dispatched=%s]",
            session_id[:8] if session_id else "?",
            user_id[:8] if user_id else "?",
            is_minor,
            already_dispatched,
        )

        # ── Fast exits ────────────────────────────────────────────────────────
        if not transcript:
            logger.debug("AlertAgent: skipping — empty transcript")
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={}),
            )
            return

        if already_dispatched:
            logger.debug("AlertAgent: skipping — alert already dispatched this session")
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={}),
            )
            return

        # ── Resolve thresholds from config ────────────────────────────────────
        settings = get_settings()
        threshold = (
            settings.minor_alert_threshold if is_minor else settings.alert_threshold
        )

        # ── Factor evaluation ─────────────────────────────────────────────────
        try:
            current_emotion = EmotionLabel(current_emotion_str)
        except ValueError:
            logger.debug(
                "AlertAgent: unknown emotion %r — defaulting to NEUTRAL",
                current_emotion_str,
            )
            current_emotion = EmotionLabel.NEUTRAL

        active_factors: list[str] = []

        # F1 — crisis keyword
        has_crisis, matched_crisis_kw = check_crisis_keywords(transcript, is_minor)
        if has_crisis:
            active_factors.append(AlertFactor.F1_CRISIS_KEYWORD.value)
            logger.debug("AlertAgent: F1 triggered — %s", matched_crisis_kw)

        # F2 — negative emotion with sufficient confidence
        if current_emotion in EmotionLabel.negative() and emotion_confidence > 0.5:
            active_factors.append(AlertFactor.F2_NEGATIVE_EMOTION.value)
            logger.debug(
                "AlertAgent: F2 triggered — emotion=%s confidence=%.2f",
                current_emotion.value,
                emotion_confidence,
            )

        # F3 — escalation pattern (3 consecutive negative emotions)
        recent_emotions_raw = state.get("recent_emotions", "[]")
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
        negative_labels = {e.value for e in EmotionLabel.negative()}

        if len(last_three) == 3 and all(
            label in negative_labels for label in last_three
        ):
            active_factors.append(AlertFactor.F3_ESCALATION.value)
            logger.debug("AlertAgent: F3 triggered — last_three=%s", last_three)

        # F4 — vulnerability (minors always count)
        if is_minor:
            active_factors.append(AlertFactor.F4_VULNERABILITY.value)

        # F5 — explicit statement of intent
        has_explicit, matched_explicit_kw = check_explicit_statement(transcript)
        if has_explicit:
            active_factors.append(AlertFactor.F5_EXPLICIT_STATEMENT.value)
            logger.debug("AlertAgent: F5 triggered — %s", matched_explicit_kw)

        # De-duplicate while preserving order (dict.fromkeys idiom).
        active_factors = list(dict.fromkeys(active_factors))
        active_count = len(active_factors)

        # ── Level computation ─────────────────────────────────────────────────
        alert_level = _compute_alert_level(active_count, threshold=threshold)

        logger.info(
            "Alert factors evaluated [session=%s user=%s level=%s count=%d/%d "
            "factors=%s is_minor=%s]",
            session_id[:8] if session_id else "?",
            user_id[:8] if user_id else "?",
            alert_level.value,
            active_count,
            threshold,
            active_factors,
            is_minor,
        )

        # ── Dispatch if threshold met ──────────────────────────────────────────
        dispatch_succeeded: bool | None = None

        if alert_level in (AlertLevel.HIGH, AlertLevel.CRISIS):
            dispatch_succeeded = await self._dispatch_alert(
                ctx=ctx,
                active_factors=active_factors,
                alert_level=alert_level,
                session_id=session_id,
                user_id=user_id,
            )

        # ── Write state delta ─────────────────────────────────────────────────
        state_delta: dict[str, Any] = {
            "alert_level": alert_level.value,
            "alert_active_count": active_count,  # int, not str
            "alert_factors": active_factors,
        }

        if dispatch_succeeded is True:
            state_delta["alert_dispatched"] = "true"
            state_delta["alert_dispatch_status"] = "ok"
        elif dispatch_succeeded is False:
            # Mark dispatched even on failure — we don't want to retry within
            # the same session and potentially spam a failed guardian lookup.
            state_delta["alert_dispatched"] = "true"
            state_delta["alert_dispatch_status"] = "failed"

        yield Event(
            author=self.name,
            actions=EventActions(state_delta=state_delta),
        )

    # ── Alert dispatch ────────────────────────────────────────────────────────

    async def _dispatch_alert(
        self,
        ctx: InvocationContext,
        active_factors: list[str],
        alert_level: AlertLevel,
        session_id: str,
        user_id: str,
    ) -> bool:
        """Dispatch guardian SMS and persist the alert record.

        Returns:
            True  — SMS sent and DB record persisted successfully.
            False — Some or all of the dispatch failed (already logged).

        Never raises — alert dispatch failures must not crash the pipeline.
        The caller always marks the session as dispatched regardless of the
        return value, to prevent retry spam on the same session.
        """
        sms_sent = False
        guardian_notified_at: str | None = None
        dispatch_succeeded = False

        try:
            client = await get_service_client()

            # ── Fetch guardian links ──────────────────────────────────────────
            guardian_result = await asyncio.wait_for(
                client.table("guardian_links")
                .select("guardian_user_id")
                .eq("patient_user_id", user_id)
                .eq("can_view_alerts", True)
                .execute(),
                timeout=_DB_TIMEOUT_SECONDS,
            )

            # ── Fetch patient display name ────────────────────────────────────
            profile_result = await asyncio.wait_for(
                client.table("user_profiles")
                .select("display_name")
                .eq("user_id", user_id)
                .maybe_single()
                .execute(),
                timeout=_DB_TIMEOUT_SECONDS,
            )

            profile_data = profile_result.data if profile_result is not None else None
            display_name: str = (
                str(profile_data.get("display_name") or "your patient")
                if isinstance(profile_data, dict)
                else "your patient"
            )

            # ── Send SMS to each guardian ─────────────────────────────────────
            guardian_data = (
                guardian_result.data if guardian_result is not None else []
            ) or []

            for link in guardian_data:
                if not isinstance(link, dict):
                    continue

                guardian_id = link.get("guardian_user_id")
                if not guardian_id:
                    continue

                try:
                    guardian_auth = await asyncio.wait_for(
                        client.auth.admin.get_user_by_id(str(guardian_id)),
                        timeout=_AUTH_LOOKUP_TIMEOUT_SECONDS,
                    )
                    user_metadata = (
                        guardian_auth.user.user_metadata
                        if guardian_auth.user is not None
                        else None
                    )
                    phone: str | None = (
                        str(user_metadata["phone"]).strip()
                        if isinstance(user_metadata, dict)
                        and user_metadata.get("phone")
                        else None
                    )

                    if not phone:
                        logger.warning(
                            "AlertAgent: guardian phone missing [guardian=%s]",
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

                except asyncio.TimeoutError:
                    logger.error(
                        "AlertAgent: guardian auth lookup timed out [guardian=%s]",
                        str(guardian_id)[:8],
                    )
                except Exception as exc:
                    logger.error(
                        "AlertAgent: guardian SMS failed [guardian=%s]: %s",
                        str(guardian_id)[:8],
                        exc,
                    )

            # ── Persist the alert record ──────────────────────────────────────
            await asyncio.wait_for(
                client.table("alerts")
                .insert(
                    {
                        "session_id": session_id or None,
                        "user_id": user_id,
                        "alert_level": alert_level.value,
                        "alert_factors": active_factors,
                        "notified_guardian": sms_sent,
                        "guardian_notified_at": guardian_notified_at,
                        "sms_sent": sms_sent,
                    }
                )
                .execute(),
                timeout=_DB_TIMEOUT_SECONDS,
            )

            ctx.session.state["post_alert_message"] = get_post_alert_message()

            logger.warning(
                "ALERT DISPATCHED [session=%s user=%s level=%s factors=%s sms_sent=%s]",
                session_id[:8] if session_id else "?",
                user_id[:8] if user_id else "?",
                alert_level.value,
                active_factors,
                sms_sent,
            )

            dispatch_succeeded = True

        except asyncio.TimeoutError:
            logger.error(
                "AlertAgent: dispatch timed out [session=%s user=%s level=%s]",
                session_id[:8] if session_id else "?",
                user_id[:8] if user_id else "?",
                alert_level.value,
            )
            dispatch_succeeded = False
            await self._persist_fallback_alert(
                alert_level=alert_level,
                active_factors=active_factors,
                session_id=session_id,
                user_id=user_id,
            )
            ctx.session.state["post_alert_message"] = get_post_alert_message()

        except Exception as exc:
            logger.error(
                "AlertAgent: dispatch failed [session=%s user=%s level=%s]: %s",
                session_id[:8] if session_id else "?",
                user_id[:8] if user_id else "?",
                alert_level.value,
                exc,
            )
            dispatch_succeeded = False
            await self._persist_fallback_alert(
                alert_level=alert_level,
                active_factors=active_factors,
                session_id=session_id,
                user_id=user_id,
            )
            ctx.session.state["post_alert_message"] = get_post_alert_message()

        return dispatch_succeeded

    async def _persist_fallback_alert(
        self,
        *,
        alert_level: AlertLevel,
        active_factors: list[str],
        session_id: str,
        user_id: str,
    ) -> None:
        """Best-effort: write an alert record even when the main dispatch path failed.

        This ensures the alert is visible in the guardian dashboard even if
        the SMS send failed, so a human can follow up manually.
        Never raises — a double-failure is logged and silently swallowed.
        """
        try:
            client = await get_service_client()
            await asyncio.wait_for(
                client.table("alerts")
                .insert(
                    {
                        "session_id": session_id or None,
                        "user_id": user_id,
                        "alert_level": alert_level.value,
                        "alert_factors": active_factors,
                        "notified_guardian": False,
                        "guardian_notified_at": None,
                        "sms_sent": False,
                    }
                )
                .execute(),
                timeout=_DB_TIMEOUT_SECONDS,
            )
            logger.info(
                "AlertAgent: fallback alert record persisted [session=%s user=%s]",
                session_id[:8] if session_id else "?",
                user_id[:8] if user_id else "?",
            )
        except Exception as db_exc:
            logger.error(
                "AlertAgent: fallback alert record also failed [session=%s user=%s]: %s",
                session_id[:8] if session_id else "?",
                user_id[:8] if user_id else "?",
                db_exc,
            )


# ── Singleton ─────────────────────────────────────────────────────────────────

alert_agent = AlertAgent(name="alert_agent")
