"""BEAN AI — Task Agent.

Handles reminders, calendar events, and task management.
Uses Gemini Flash (cheap tier) for task parsing and responses.

Pipeline per turn:
    1. Parse user intent (create/list/delete reminder)
    2. Interact with calendar_service or Supabase tasks table
    3. Confirm action back to user
"""

import logging
from collections.abc import AsyncGenerator
from datetime import UTC, datetime

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from services.calendar_service import CalendarService, get_calendar_token
from services.llm_service import generate_json
from services.supabase_client import get_service_client

logger = logging.getLogger(__name__)

TASK_SYSTEM = """You are BEAN's task management assistant.
Parse the user's request and extract task/reminder details.

Respond ONLY with this exact JSON — no markdown, no preamble:
{
  "action": "create|list|delete|unknown",
  "title": "task title if creating",
  "due_at": "ISO 8601 datetime string or null",
  "task_id": "task ID if deleting or null"
}"""


class TaskAgent(BaseAgent):
    """Task management agent — reminders and calendar integration."""

    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        transcript = ctx.session.state.get("current_transcript", "")
        user_id = ctx.session.state.get("user_id", "")

        if not transcript:
            response = "What would you like me to help you remember?"
            ctx.session.state["response_text"] = response
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"response_text": response}),
            )
            return

        # ── Parse intent with LLM ──
        try:
            parsed = await generate_json(
                task="task_management",
                prompt=f"User said: {transcript}",
                system=TASK_SYSTEM,
            )
        except Exception as exc:
            logger.error("Task parsing failed: %s", exc)
            response = "I had trouble understanding that. Could you try again?"
            ctx.session.state["response_text"] = response
            yield Event(
                author=self.name,
                actions=EventActions(state_delta={"response_text": response}),
            )
            return

        action = parsed.get("action", "unknown")
        response = await self._handle_action(action, parsed, user_id)

        ctx.session.state["response_text"] = response
        logger.info(
            "TaskAgent: action=%s user=%s", action, user_id[:8] if user_id else ""
        )

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={"response_text": response}),
        )

    async def _handle_action(self, action: str, parsed: dict, user_id: str) -> str:
        """Handle create, list, delete or unknown task actions."""

        if action == "create":
            return await self._create_task(parsed, user_id)
        elif action == "list":
            return await self._list_tasks(user_id)
        elif action == "delete":
            return await self._delete_task(parsed, user_id)
        else:
            return "I can help you create reminders, list your tasks, or delete them. What would you like?"

    async def _create_task(self, parsed: dict, user_id: str) -> str:
        """Create a task in Supabase and optionally in Google Calendar."""
        title = parsed.get("title", "Reminder")
        due_at_str = parsed.get("due_at")

        try:
            due_at = datetime.fromisoformat(due_at_str) if due_at_str else None
        except (ValueError, TypeError):
            due_at = None

        try:
            client = await get_service_client()
            await client.table("tasks").insert(
                {
                    "user_id": user_id,
                    "title": title,
                    "due_at": due_at.isoformat() if due_at else None,
                    "status": "pending",
                    "created_at": datetime.now(UTC).isoformat(),
                }
            ).execute()

            # Try Google Calendar if token available
            token = await get_calendar_token(user_id)
            if token and due_at:
                try:
                    cal = CalendarService(token)
                    cal_id = await cal.find_or_create_bean_calendar()
                    await cal.create_event(
                        calendar_id=cal_id,
                        title=title,
                        event_time=due_at,
                    )
                except Exception as exc:
                    logger.warning("Calendar sync failed (non-fatal): %s", exc)

            if due_at:
                return f"Got it! I'll remind you about '{title}' on {due_at.strftime('%B %d at %I:%M %p')}."
            else:
                return f"Added '{title}' to your tasks!"

        except Exception as exc:
            logger.error("Task creation failed: %s", exc)
            return "I couldn't save that reminder. Please try again."

    async def _list_tasks(self, user_id: str) -> str:
        """List pending tasks for the user."""
        try:
            client = await get_service_client()
            result = (
                await client.table("tasks")
                .select("title, due_at, status")
                .eq("user_id", user_id)
                .eq("status", "pending")
                .order("due_at")
                .limit(5)
                .execute()
            )

            if not result.data:
                return "You have no pending tasks right now!"

            items = []
            for task in result.data:
                title = task.get("title", "")
                due_at = task.get("due_at")
                if due_at:
                    try:
                        dt = datetime.fromisoformat(due_at)
                        items.append(f"• {title} — {dt.strftime('%b %d')}")
                    except (ValueError, TypeError):
                        items.append(f"• {title}")
                else:
                    items.append(f"• {title}")

            return "Here are your tasks:\n" + "\n".join(items)

        except Exception as exc:
            logger.error("Task list failed: %s", exc)
            return "I couldn't retrieve your tasks right now."

    async def _delete_task(self, parsed: dict, user_id: str) -> str:
        """Delete a task by ID."""
        task_id = parsed.get("task_id")
        if not task_id:
            return "Which task would you like to delete?"

        try:
            client = await get_service_client()
            await client.table("tasks").delete().eq("id", task_id).eq(
                "user_id", user_id
            ).execute()
            return "Done! I've removed that task."
        except Exception as exc:
            logger.error("Task delete failed: %s", exc)
            return "I couldn't delete that task. Please try again."


# ── Singleton ─────────────────────────────────────────────────────────────────
task_agent = TaskAgent(name="task_agent")
