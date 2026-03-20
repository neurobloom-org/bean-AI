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
from datetime import UTC, datetime

from services.supabase_client import get_service_client
from services.twilio_service import send_sms

logger = logging.getLogger(__name__)


async def check_pending_reminders() -> int:
    """Send SMS reminders for due tasks."""
    try:
        client = await get_service_client()
        now = datetime.now(UTC).isoformat()

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
            task_id = task.get("id")
            title = task.get("title", "Task reminder")
            description = task.get("description")
            phone_number = task.get("phone_number")

            if not task_id or not phone_number:
                logger.warning(
                    "[Reminder] Skipping task with missing id or phone_number"
                )
                continue

            message = f"Reminder: {title}"
            if description:
                message += f" - {description}"

            try:
                await send_sms(
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