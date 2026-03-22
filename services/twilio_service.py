"""BEAN AI v5 — Twilio SMS service.

Privacy: SMS messages contain ONLY generic concern notices.
No conversation content, no trigger phrases, no diagnostic info.

Exported functions:
    send_sms(to, body)          — general-purpose SMS sender (base function)
    send_guardian_sms(to, body) — backward-compat alias for send_sms
    send_guardian_alert(...)    — structured guardian alert with retry (used by AlertAgent)
    _reset_client()             — test helper to clear the cached Twilio client

Design notes:
    - All Twilio calls run via asyncio.to_thread() because the Twilio SDK is
      synchronous. Each call is wrapped in asyncio.wait_for() to prevent a
      hanging Twilio connection from blocking the thread pool indefinitely.
    - send_guardian_alert retries once on transient failures because guardian
      notifications are safety-critical. send_sms does not retry — reminder
      SMS failures are logged and retried on the next background job cycle.
    - Body length is capped at _MAX_SMS_BODY_CHARS. Twilio accepts up to 1600
      characters but charges per 160-char segment; we truncate early to avoid
      runaway costs and unexpected multi-part splits.
"""

from __future__ import annotations

import asyncio
import logging
import re

from twilio.rest import Client

from shared.config import get_settings

__all__ = [
    "send_sms",
    "send_guardian_sms",
    "send_guardian_alert",
    "_reset_client",
]

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Twilio hard limit is 1600 chars; we cap lower to avoid surprise multi-part costs.
_MAX_SMS_BODY_CHARS: int = 320

# Timeout for a single Twilio API call.
_TWILIO_CALL_TIMEOUT_SECONDS: float = 10.0

# How many times to retry a failed guardian alert before giving up.
# Reminder SMS (send_sms) does NOT retry — the background job handles that.
_GUARDIAN_ALERT_MAX_ATTEMPTS: int = 2
_GUARDIAN_ALERT_RETRY_DELAY_SECONDS: float = 1.5

# Minimal E.164 pattern: starts with +, followed by 7-15 digits only.
_E164_RE = re.compile(r"^\+\d{7,15}$")

# ── Singleton client ──────────────────────────────────────────────────────────

_client: Client | None = None


def _get_client() -> Client:
    """Return a cached Twilio Client, creating one on first call."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    return _client


def _reset_client() -> None:
    """Clear the cached Twilio client.

    Use in tests to force re-initialisation after patching settings, or to
    simulate credential rotation without restarting the process.
    """
    global _client
    _client = None


# ── Validation helpers ────────────────────────────────────────────────────────


def _is_configured() -> tuple[bool, str]:
    """Return (ok, reason) indicating whether Twilio is fully configured."""
    settings = get_settings()
    if not settings.twilio_account_sid:
        return False, "TWILIO_ACCOUNT_SID is not set"
    if not settings.twilio_auth_token:
        return False, "TWILIO_AUTH_TOKEN is not set"
    if not settings.twilio_from_number:
        return False, "TWILIO_FROM_NUMBER is not set"
    return True, ""


def _validate_phone(number: str) -> str | None:
    """Return a normalised E.164 number, or None if the format looks invalid.

    This is a lightweight regex check, not a Twilio number lookup. It catches
    the most common mistake (missing country code) before burning an API call.
    """
    stripped = number.strip()
    if not stripped:
        return None
    if not _E164_RE.match(stripped):
        return None
    return stripped


def _truncate_body(body: str) -> str:
    """Truncate an SMS body to _MAX_SMS_BODY_CHARS, appending '…' if cut."""
    if len(body) <= _MAX_SMS_BODY_CHARS:
        return body
    return body[: _MAX_SMS_BODY_CHARS - 1] + "…"


# ── Core send function ────────────────────────────────────────────────────────


async def send_sms(to: str, body: str) -> bool:
    """Send an SMS to any phone number.

    This is the base function. All other SMS helpers delegate here.

    Args:
        to:   Recipient phone in E.164 format (e.g. +94771234567).
              Numbers that don't pass the E.164 check are rejected before
              reaching Twilio so we don't burn API credits on bad numbers.
        body: Message text. Truncated to _MAX_SMS_BODY_CHARS automatically.
              Keep content privacy-safe — no conversation text, no triggers.

    Returns:
        True on success, False on any failure (already logged).
        Does NOT retry — callers that need retry should call send_guardian_alert
        or handle retries themselves.
    """
    configured, reason = _is_configured()
    if not configured:
        logger.warning(
            "Twilio not configured (%s) — SMS not sent to %s***",
            reason,
            to[:6],
        )
        return False

    phone = _validate_phone(to)
    if phone is None:
        logger.warning(
            "send_sms: invalid phone number format %r (expected E.164 like +94771234567)",
            to[:10] + "***",
        )
        return False

    if not body or not body.strip():
        logger.warning("send_sms: empty body — skipping send to %s***", phone[:6])
        return False

    truncated_body = _truncate_body(body.strip())
    settings = get_settings()

    try:
        client = _get_client()
        message = await asyncio.wait_for(
            asyncio.to_thread(
                client.messages.create,
                body=truncated_body,
                from_=settings.twilio_from_number,
                to=phone,
            ),
            timeout=_TWILIO_CALL_TIMEOUT_SECONDS,
        )
        logger.info("SMS sent: sid=%s to=%s***", message.sid, phone[:6])
        return True

    except asyncio.TimeoutError:
        logger.error(
            "send_sms: Twilio call timed out after %.1fs for %s***",
            _TWILIO_CALL_TIMEOUT_SECONDS,
            phone[:6],
        )
        return False

    except Exception as exc:
        logger.error("send_sms: failed to %s***: %s", phone[:6], exc)
        return False


# ── Backward-compat alias ─────────────────────────────────────────────────────


async def send_guardian_sms(to: str, body: str) -> bool:
    """Send an SMS to a guardian/doctor phone number.

    Thin alias over send_sms() kept for backward compatibility with any
    callers that already import this name directly.
    """
    return await send_sms(to=to, body=body)


# ── Guardian alert (safety-critical, with retry) ──────────────────────────────


async def send_guardian_alert(
    guardian_phone: str,
    user_display_name: str,
    alert_level: str,
    session_id: str,
) -> str:
    """Send a structured guardian alert SMS and return the Twilio message SID.

    Used exclusively by AlertAgent. This function:
      - Builds a privacy-safe message body (no conversation content).
      - Retries once on transient failure (_GUARDIAN_ALERT_MAX_ATTEMPTS).
      - Raises RuntimeError if Twilio is not configured.
      - Raises RuntimeError if all retry attempts are exhausted.

    Args:
        guardian_phone:    Guardian's E.164 phone number.
        user_display_name: Patient display name (never raw transcript text).
        alert_level:       One of: "low", "medium", "high", "crisis".
        session_id:        Session ID for internal log correlation only.
                           Never included in the SMS body.

    Returns:
        Twilio message SID on success.

    Raises:
        RuntimeError: Twilio not configured, or all send attempts failed.
        ValueError:   Invalid guardian phone number format.
    """
    configured, reason = _is_configured()
    if not configured:
        raise RuntimeError(f"Twilio not configured — {reason}")

    phone = _validate_phone(guardian_phone)
    if phone is None:
        raise ValueError(
            f"send_guardian_alert: invalid phone number format "
            f"{guardian_phone[:10]!r} (expected E.164 like +94771234567)"
        )

    # Build a privacy-safe body — no conversation content, no trigger phrases.
    safe_name = (user_display_name or "your patient").strip()[:80]
    if alert_level == "crisis":
        body = (
            f"⚠️ BEAN AI URGENT: {safe_name} may need immediate support. "
            "Please check in with them now or contact emergency services if needed."
        )
    else:
        body = (
            f"BEAN AI: {safe_name} may benefit from extra support today. "
            "Consider checking in when you can."
        )

    body = _truncate_body(body)
    settings = get_settings()
    last_exc: Exception | None = None

    for attempt in range(1, _GUARDIAN_ALERT_MAX_ATTEMPTS + 1):
        try:
            client = _get_client()
            message = await asyncio.wait_for(
                asyncio.to_thread(
                    client.messages.create,
                    body=body,
                    from_=settings.twilio_from_number,
                    to=phone,
                ),
                timeout=_TWILIO_CALL_TIMEOUT_SECONDS,
            )

            # Log at WARNING so alert sends surface in production dashboards.
            logger.warning(
                "Guardian alert SMS sent: sid=%s to=%s*** level=%s session=%s",
                message.sid,
                phone[:6],
                alert_level,
                session_id[:8] if session_id else "?",
            )
            return str(message.sid)

        except asyncio.TimeoutError as exc:
            last_exc = exc
            logger.error(
                "send_guardian_alert: attempt %d/%d timed out after %.1fs "
                "(level=%s session=%s)",
                attempt,
                _GUARDIAN_ALERT_MAX_ATTEMPTS,
                _TWILIO_CALL_TIMEOUT_SECONDS,
                alert_level,
                session_id[:8] if session_id else "?",
            )

        except Exception as exc:
            last_exc = exc
            logger.error(
                "send_guardian_alert: attempt %d/%d failed: %s "
                "(level=%s session=%s)",
                attempt,
                _GUARDIAN_ALERT_MAX_ATTEMPTS,
                exc,
                alert_level,
                session_id[:8] if session_id else "?",
            )

        if attempt < _GUARDIAN_ALERT_MAX_ATTEMPTS:
            await asyncio.sleep(_GUARDIAN_ALERT_RETRY_DELAY_SECONDS)

    raise RuntimeError(
        f"send_guardian_alert: all {_GUARDIAN_ALERT_MAX_ATTEMPTS} attempts failed "
        f"for {phone[:6]}*** (level={alert_level})"
    ) from last_exc