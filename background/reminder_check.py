"""BEAN AI — Reminder SMS checker background job.

Runs every `settings.reminder_check_interval_seconds` (default: 60s).

Finds tasks where:
    status IN ('pending', 'snoozed')
    reminder_at <= NOW()
    reminder_at >= NOW() - _STALENESS_CUTOFF_HOURS  (skip very stale reminders)

Then for each task:
    1. Looks up the task owner's phone number from Supabase auth metadata.
       The `tasks` table has no phone_number column — the phone lives in
       auth.users.user_metadata (same pattern used by the guardian alert system).
    2. Delivers reminder into the active conversation if the robot is online.
    3. Falls back to SMS only when no active session exists.
    4. Marks delivery and schedules a 30-minute followup for ignored reminders.

Concurrency:
    SMS sends run concurrently up to _MAX_CONCURRENT_SMS at a time using
    an asyncio.Semaphore. Phone lookups are batched by unique user_id so
    we hit auth.admin.get_user_by_id() at most once per user per cycle.

Staleness guard:
    If reminder_at was more than _STALENESS_CUTOFF_HOURS ago, the task is
    skipped and logged. This prevents a server restart from firing a flood
    of stale reminders that should have gone out hours earlier.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from services.redis_service import redis_set
from services.supabase_client import get_service_client
from services.twilio_service import send_sms
from shared.config import get_settings

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Tasks with reminder_at older than this are skipped — they're too stale to
# be useful and firing them now would confuse the user.
_STALENESS_CUTOFF_HOURS: int = 2

# Max tasks fetched per cycle. Keeps each run fast and within Twilio rate limits.
_TASK_BATCH_LIMIT: int = 50

# Max concurrent Twilio calls per cycle. Twilio's free tier allows ~1 req/s;
# paid accounts have much higher limits. Keep this conservative by default.
_MAX_CONCURRENT_SMS: int = 5

# Max chars for the task title in the SMS body. Titles can be up to 200 chars
# (per TaskCreate schema) — truncate so the full body stays readable.
_MAX_TITLE_CHARS: int = 80

# Max chars for the description snippet in the SMS body.
_MAX_DESC_CHARS: int = 100


# ── Phone number resolution ───────────────────────────────────────────────────


async def _fetch_user_phones(
    user_ids: list[str],
) -> dict[str, str]:
    """Look up phone numbers for a list of user IDs from Supabase auth metadata.

    The tasks table has no phone_number column. Phones are stored in
    auth.users.user_metadata (set during signup or profile update), which is
    the same place the AlertAgent reads guardian phones from.

    Requires the service-role client (admin.get_user_by_id is admin-only).

    Args:
        user_ids: List of unique user UUIDs to look up.

    Returns:
        Dict mapping user_id -> E.164 phone number for users where a phone
        was found. Users with no phone number are simply absent from the dict.
    """
    if not user_ids:
        return {}

    client = await get_service_client()
    phone_map: dict[str, str] = {}

    for user_id in user_ids:
        try:
            auth_result = await client.auth.admin.get_user_by_id(user_id)
            user = auth_result.user if auth_result else None
            if user is None:
                logger.warning(
                    "[Reminder] Auth user not found for user_id=%s", user_id[:8]
                )
                continue

            metadata = user.user_metadata or {}
            phone = metadata.get("phone") or metadata.get("phone_number")

            if isinstance(phone, str) and phone.strip():
                phone_map[user_id] = phone.strip()
            else:
                logger.debug(
                    "[Reminder] No phone in user_metadata for user_id=%s", user_id[:8]
                )

        except Exception as exc:
            logger.warning(
                "[Reminder] Failed to fetch auth user %s: %s", user_id[:8], exc
            )

    return phone_map


# ── SMS body builder ──────────────────────────────────────────────────────────


def _build_reminder_body(title: str, description: str | None) -> str:
    """Build a concise SMS reminder body.

    Truncates title and description independently so each gets fair space,
    and appends '…' when truncated so the user knows text was cut.
    """
    safe_title = (title or "Task reminder").strip()
    if len(safe_title) > _MAX_TITLE_CHARS:
        safe_title = safe_title[: _MAX_TITLE_CHARS - 1] + "…"

    body = f"BEAN Reminder: {safe_title}"

    if isinstance(description, str) and description.strip():
        safe_desc = description.strip()
        if len(safe_desc) > _MAX_DESC_CHARS:
            safe_desc = safe_desc[: _MAX_DESC_CHARS - 1] + "…"
        body += f" — {safe_desc}"

    return body


# ── Per-task send + mark ──────────────────────────────────────────────────────


async def _send_and_mark(
    task: Mapping,  # type: ignore[type-arg]
    phone: str,
    sem: asyncio.Semaphore,
) -> None:
    """Send a reminder SMS for one task and mark it as 'reminded' on success.

    Uses the semaphore to cap concurrent Twilio calls. Status update uses
    an extra `.eq("status", "pending")` guard so a concurrent update (e.g.
    the user completing the task between the query and the update) is not
    silently clobbered.
    """
    task_id = task.get("id")
    title = task.get("title") or "Task reminder"
    description = task.get("description")

    body = _build_reminder_body(str(title), description)

    async with sem:
        sent = await send_sms(to=phone, body=body)

    if sent:
        try:
            client = await get_service_client()
            await (
                client.table("tasks")
                .update(
                    {
                        "status": "reminded",
                        "updated_at": datetime.now(UTC).isoformat(),
                    }
                )
                .eq("id", task_id)
                .in_("status", ["pending", "snoozed"])
                .execute()
            )
            logger.info(
                "[Reminder] Sent and marked task %s as reminded",
                str(task_id)[:8],
            )
        except Exception as exc:
            logger.error(
                "[Reminder] SMS sent but failed to mark task %s: %s",
                str(task_id)[:8],
                exc,
            )
    else:
        logger.warning(
            "[Reminder] SMS failed for task %s — will retry next cycle",
            str(task_id)[:8],
        )


# ── Main job function ─────────────────────────────────────────────────────────


async def check_pending_reminders() -> int:
    """Deliver reminders in-conversation when possible, else fall back to SMS.

    Returns:
        Number of SMS fallback messages successfully scheduled this run.
    """
    now = datetime.now(UTC)
    staleness_cutoff = (now - timedelta(hours=_STALENESS_CUTOFF_HOURS)).isoformat()
    now_iso = now.isoformat()

    try:
        client = await get_service_client()

        result = (
            await client.table("tasks")
            .select("id, user_id, title, description, reminder_at, due_at, status")
            .in_("status", ["pending", "snoozed"])
            .not_.is_("reminder_at", "null")
            .lte("reminder_at", now_iso)
            .gte("reminder_at", staleness_cutoff)
            .order("reminder_at", desc=False)
            .limit(_TASK_BATCH_LIMIT)
            .execute()
        )

        tasks = result.data or []

        # Batch phone lookups — one auth call per unique user, not per task.
        unique_user_ids = list(
            {
                str(t["user_id"])
                for t in tasks
                if isinstance(t, Mapping) and t.get("user_id")
            }
        )

        phone_map = await _fetch_user_phones(unique_user_ids) if unique_user_ids else {}

        # Import here to avoid circular import at module level.
        # _active_sessions is the live registry of connected WebSocket sessions.
        from api.websocket_handler import _active_sessions  # noqa: PLC0415

        sem = asyncio.Semaphore(_MAX_CONCURRENT_SMS)
        send_tasks: list[asyncio.Task[None]] = []

        for task in tasks:
            if not isinstance(task, Mapping):
                continue
            task_id = task.get("id")
            user_id = str(task.get("user_id") or "")
            if not task_id or not user_id:
                continue

            # Write reminder to Redis so the WebSocket handler's
            # _poll_reminders loop can pick it up for in-conversation delivery.
            redis_key = f"pending_reminder:{user_id}"
            await redis_set(
                redis_key,
                {
                    "task_id": str(task_id),
                    "title": task.get("title"),
                    "description": task.get("description"),
                    "due_at": task.get("due_at"),
                },
                ttl_seconds=3600,  # 1h — if robot never comes online, expire
            )

            # Check whether the user has an active WebSocket session right now.
            # If they do, the _poll_reminders loop will pick up the Redis key
            # within 15 s, so no SMS is needed.
            # If they don't, fall back to SMS immediately so the reminder is
            # not silently lost when the Redis TTL expires.
            user_has_active_session = any(
                str(s.get("user_id", "")) == user_id
                for s in _active_sessions.values()
            )

            if user_has_active_session:
                # In-conversation delivery — mark the task and schedule followup.
                followup_at = (datetime.now(UTC) + timedelta(minutes=30)).isoformat()
                try:
                    await (
                        client.table("tasks")
                        .update({
                            "reminder_delivered_at": datetime.now(UTC).isoformat(),
                            "followup_reminder_at": followup_at,
                            "status": "reminded",
                        })
                        .eq("id", task_id)
                        .in_("status", ["pending", "snoozed"])
                        .execute()
                    )
                except Exception as exc:
                    logger.warning("[Reminder] Failed to mark task delivered: %s", exc)

                logger.info(
                    "[Reminder] Queued in-conversation reminder for user %s task %s",
                    user_id[:8],
                    str(task_id)[:8],
                )

            else:
                # Robot is offline — SMS fallback so the user actually gets notified.
                phone = phone_map.get(user_id)
                if phone:
                    send_tasks.append(
                        asyncio.create_task(
                            _send_and_mark(task, phone, sem),
                            name=f"sms-reminder-{str(task_id)[:8]}",
                        )
                    )
                    logger.info(
                        "[Reminder] No active session for user %s — queuing SMS for task %s",
                        user_id[:8],
                        str(task_id)[:8],
                    )
                else:
                    logger.warning(
                        "[Reminder] No active session and no phone for user %s task %s — reminder lost",
                        user_id[:8],
                        str(task_id)[:8],
                    )

        # ── Followup pass: tasks where first reminder was delivered but not acknowledged ──
        followup_result = await (
            client.table("tasks")
            .select("id, user_id, title, description, due_at")
            .in_("status", ["pending", "reminded"])
            .eq("reminder_acknowledged", False)
            .not_.is_("followup_reminder_at", "null")
            .lte("followup_reminder_at", now_iso)
            .limit(50)
            .execute()
        )

        for task in (followup_result.data or []):
            if not isinstance(task, Mapping):
                logger.warning("[Reminder] Skipping non-mapping followup task row")
                continue

            user_id = str(task.get("user_id") or "")
            if not user_id:
                continue

            redis_key = f"pending_reminder:{user_id}"
            await redis_set(
                redis_key,
                {
                    "task_id": str(task.get("id")),
                    "title": task.get("title"),
                    "description": task.get("description"),
                    "due_at": task.get("due_at"),
                },
                ttl_seconds=3600,
            )
            # Clear followup so it doesn't fire again
            await (
                client.table("tasks")
                .update({"followup_reminder_at": None})
                .eq("id", task.get("id"))
                .execute()
            )

        if not send_tasks:
            return 0

        results = await asyncio.gather(*send_tasks, return_exceptions=True)

        errors = sum(1 for r in results if isinstance(r, BaseException))
        sent = len(send_tasks) - errors

        if errors:
            logger.error(
                "[Reminder] %d send task(s) raised unexpected exceptions this cycle",
                errors,
            )

        return sent

    except Exception as exc:
        logger.error("[Reminder] Job failed: %s", exc)
        return 0


# ── Background loop ───────────────────────────────────────────────────────────


async def run_reminder_check_loop() -> None:
    """Run the reminder check on a configurable interval.

    Interval is read from settings.reminder_check_interval_seconds (default 60s)
    so it can be tuned without a code change.

    Exits cleanly on asyncio.CancelledError (triggered by the FastAPI lifespan
    shutdown). All other exceptions are caught so a single bad cycle never
    kills the loop.
    """
    settings = get_settings()
    interval = settings.reminder_check_interval_seconds

    logger.info("[Reminder] Loop started — interval=%ds", interval)

    while True:
        try:
            count = await check_pending_reminders()

            if count:
                logger.info(
                    "[Reminder] Sent %d reminder SMS message(s) this cycle", count
                )

            await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.info("[Reminder] Loop cancelled — shutting down cleanly")
            break

        except Exception as exc:
            logger.error("[Reminder] Loop error (will retry next cycle): %s", exc)
            await asyncio.sleep(interval)