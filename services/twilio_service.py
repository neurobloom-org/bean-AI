"""BEAN AI v5 — Twilio SMS service.

Privacy: SMS messages contain ONLY generic concern notices.
No conversation content, no trigger phrases, no diagnostic info.
"""

import logging

from twilio.rest import Client

from shared.config import get_settings

logger = logging.getLogger(__name__)
_client: Client | None = None


def _get_client() -> Client:
    global _client
    if _client is None:
        settings = get_settings()
        _client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    return _client


async def send_guardian_sms(to: str, body: str) -> bool:
    """Send an SMS to a guardian/doctor phone number.

    Args:
        to:   Guardian's phone number (E.164 format, e.g. +94771234567)
        body: Privacy-safe message body (no conversation content)

    Returns:
        True on success, False on failure.
    """
    settings = get_settings()
    if not settings.twilio_account_sid or not settings.twilio_auth_token:
        logger.warning("Twilio not configured — SMS not sent to %s", to[:6] + "***")
        return False

    try:
        import asyncio

        client = _get_client()
        message = await asyncio.to_thread(
            client.messages.create,
            body=body,
            from_=settings.twilio_from_number,
            to=to,
        )
        logger.info("SMS sent: sid=%s to=%s***", message.sid, to[:6])
        return True
    except Exception as exc:
        logger.error("SMS send failed to %s***: %s", to[:6], exc)
        return False


async def send_guardian_alert(
    guardian_phone: str,
    user_display_name: str,
    alert_level: str,
    session_id: str,
) -> str:
    """Convenience wrapper used by AlertAgent.

    Returns the Twilio message SID on success, raises on failure.
    """
    if alert_level == "crisis":
        body = (
            f"⚠️ BEAN AI URGENT: {user_display_name} may need immediate support. "
            f"Please check in with them now or contact emergency services if needed."
        )
    else:
        body = (
            f"BEAN AI: {user_display_name} may benefit from extra support today. "
            f"Consider checking in when you can."
        )

    settings = get_settings()
    if not settings.twilio_account_sid or not settings.twilio_auth_token:
        raise RuntimeError("Twilio not configured")

    import asyncio

    client = _get_client()
    message = await asyncio.to_thread(
        client.messages.create,
        body=body,
        from_=settings.twilio_from_number,
        to=guardian_phone,
    )
    return message.sid
