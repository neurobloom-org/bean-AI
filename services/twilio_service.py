"""
services/twilio_service.py
===========================
Twilio SMS service for crisis alerts.
Only needs shared/ — no other services needed.
"""

from twilio.rest import Client

from shared.config import config

twilio = Client(config.TWILIO_SID, config.TWILIO_TOKEN)


async def send_sms(to: str, message: str) -> str:
    """Send an SMS message via Twilio."""
    msg = twilio.messages.create(
        body=message,
        from_=config.TWILIO_FROM,
        to=to,
    )
    return msg.sid


async def send_crisis_alert(to: str, user_id: str) -> str:
    """Send a crisis alert SMS to a guardian."""
    message = (
        f"Bean Crisis Alert\n"
        f"User {user_id} may need immediate support.\n"
        f"Please check in with them as soon as possible."
    )
    return await send_sms(to, message)
