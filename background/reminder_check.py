"""BEAN AI — Reminder check background job.

Runs every N seconds. Queries tasks table for pending reminders,
injects them into the active WebSocket session via _active_sessions.
"""

import asyncio
import logging
from datetime import UTC, datetime

from shared.config import get_settings

logger = logging.getLogger(__name__)


async def check_pending_reminders() -> int:
    """Check for pending reminders that need to be sent.

    Finds tasks where:
      - status = 'pending' or 'snoozed'
      - reminder_at <= NOW()

    For each task:
      - if the user has an active session, queue the reminder in that session
      - then mark the task as 'reminded'
      - if the user is offline, leave it unchanged so it can be retried later
    """
    from api.websocket_handler import _active_sessions
    from services.supabase_client import get_service_client

    delivered_count = 0

    try:
        client = await get_service_client()
        now = datetime.now(UTC).isoformat()

        result = (
            await client.table("tasks")
            .select("id, user_id, title, description, due_at, reminder_at, status")
            .in_("status", ["pending", "snoozed"])
            .not_.is_("reminder_at", "null")
            .lte("reminder_at", now)
            .order("reminder_at", desc=False)
            .limit(50)
            .execute()
        )

        pending_tasks = result.data or []

        for task in pending_tasks:
            user_id = task.get("user_id")
            task_id = task.get("id")

            if not user_id or not task_id:
                logger.warning("Skipping malformed task row: %s", task)
                continue

            delivered = False

            for session_id, session_entry in list(_active_sessions.items()):
                if session_entry.get("user_id") != user_id:
                    continue

                reminder_payload = {
                    "task_id": task_id,
                    "title": task.get("title", ""),
                    "description": task.get("description"),
                    "due_at": task.get("due_at"),
                    "reminder_at": task.get("reminder_at"),
                }

                # Use a queue/list so multiple reminders do not overwrite each other.
                session_entry.setdefault("pending_reminders", []).append(reminder_payload)

                delivered = True
                logger.info(
                    "Reminder queued for delivery: task=%s user=%s session=%s",
                    task_id,
                    str(user_id)[:8],
                    str(session_id)[:8],
                )
                break

            if not delivered:
                logger.debug(
                    "User offline, reminder left pending: task=%s user=%s",
                    task_id,
                    str(user_id)[:8],
                )
                continue

            try:
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
                delivered_count += 1
            except Exception as exc:
                logger.error("Failed to mark task as reminded [%s]: %s", task_id, exc)

    except Exception as exc:
        logger.error("Reminder check failed: %s", exc)

    return delivered_count


async def run_reminder_loop() -> None:
    """Run reminder check every N seconds (configured via settings)."""
    settings = get_settings()
    interval = settings.reminder_check_interval_seconds
    logger.info("[Reminder] Loop started — interval=%ds", interval)

    while True:
        try:
            count = await check_pending_reminders()
            if count:
                logger.info("[Reminder] Queued %d reminders", count)

            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("[Reminder] Loop cancelled — shutting down")
            break
        except Exception as exc:
            logger.error("[Reminder] Loop error (will retry): %s", exc)
            await asyncio.sleep(interval)