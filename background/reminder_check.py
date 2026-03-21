"""BEAN AI — Reminder SMS checker background job.

Runs every 60 seconds.

Finds tasks where:
    status = 'pending'
    reminder_at <= NOW()

Then:
    send SMS via twilio_service
    update status -> reminded
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from importlib import import_module
from typing import Any, cast

from services.supabase_client import get_service_client

logger = logging.getLogger(__name__)

SMSFunction = Callable[..., Awaitable[Any]]


def _get_sms_sender() -> SMSFunction:
    """Load the SMS sender function from services.twilio_service.

    This avoids mypy attr-defined errors when the exact exported function
    name differs across implementations.
    """
    module = import_module("services.twilio_service")

    for function_name in (
        "send_sms",
        "send_twilio_sms",
        "send_message",
        "send_sms_message",
    ):
        candidate = getattr(module, function_name, None)
        if callable(candidate):
            return cast(SMSFunction, candidate)

    raise RuntimeError(
        "No supported SMS sender found in services.twilio_service. "
        "Expected one of: send_sms, send_twilio_sms, send_message, send_sms_message."
    )


async def check_pending_reminders() -> int:
    """Send SMS reminders for due tasks."""
    try:
        client = await get_service_client()
        now = datetime.now(UTC).isoformat()
        sms_sender = _get_sms_sender()

        sent_count = 0

        result = (
            await client.table("tasks")
            .select("id, title, description, reminder_at, phone_number")
            .eq("status", "pending")
            .not_.is_("reminder_at", "null")
            .lte("reminder_at", now)
            .limit(50)
            .execute()
        )

        pending_tasks = result.data or []

        for task in pending_tasks:
            if not isinstance(task, Mapping):
                logger.warning("[Reminder] Skipping non-mapping task row")
                continue

            task_id = task.get("id")
            title = task.get("title", "Task reminder")
            description = task.get("description")
            phone_number = task.get("phone_number")

            if not task_id or not isinstance(phone_number, str) or not phone_number:
                logger.warning(
                    "[Reminder] Skipping task with missing id or phone_number"
                )
                continue

            if not isinstance(title, str):
                title = "Task reminder"

            message = f"Reminder: {title}"
            if isinstance(description, str) and description:
                message += f" - {description}"

            try:
                await sms_sender(
                    to=phone_number,
                    body=message,
                )

                await (
                    client.table("tasks")
                    .update(
                        {
                            "status": "reminded",
                            "updated_at": datetime.now(UTC).isoformat(),
                        }
                    )
                    .eq("id", task_id)
                    .execute()
                )

                sent_count += 1

                logger.info(
                    "[Reminder] SMS reminder sent for task %s",
                    str(task_id)[:8],
                )

            except Exception as exc:
                logger.error(
                    "[Reminder] Failed sending SMS for task %s: %s",
                    str(task_id)[:8],
                    exc,
                )

        return sent_count

    except Exception as exc:
        logger.error("[Reminder] Job failed: %s", exc)
        return 0


async def run_reminder_check_loop() -> None:
    """Run reminder check every 60 seconds."""
    interval_seconds = 60

    logger.info("[Reminder] Loop started — interval=60s")

    while True:
        try:
            count = await check_pending_reminders()

            if count:
                logger.info(
                    "[Reminder] Sent %d reminder SMS messages",
                    count,
                )

            await asyncio.sleep(interval_seconds)

        except asyncio.CancelledError:
            logger.info("[Reminder] Loop cancelled")
            break

        except Exception as exc:
            logger.error("[Reminder] Loop error: %s", exc)
            await asyncio.sleep(interval_seconds)
