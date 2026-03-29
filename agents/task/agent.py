"""BEAN AI — Task Agent.

Handles reminders and task management.
Uses Gemini Flash (cheap tier) for task parsing and responses.

Pipeline per turn:
    1. If a draft is pending confirmation: check user's response (yes/correction)
    2. Otherwise: extract a new task draft from user text
    3. Ask for confirmation before writing to DB
    4. On confirmation: save to Supabase tasks table
"""

import logging
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from services.llm_service import generate_json
from services.supabase_client import get_service_client
from shared.schemas import TaskDraft

logger = logging.getLogger(__name__)

TASK_DRAFT_SYSTEM = """Extract a task draft from the user's message.

Respond ONLY with this exact JSON — no markdown, no preamble:
{
  "title": "string",
  "description": "string or null",
  "due_at": "ISO 8601 datetime string or null",
  "reminder_at": "ISO 8601 datetime string or null",
  "correction_count": 0
}

If no task can be extracted, return:
{
  "title": "",
  "description": null,
  "due_at": null,
  "reminder_at": null,
  "correction_count": 0
}"""


class TaskAgent(BaseAgent):
    """Task management agent — reminders and Supabase task storage."""

    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state
        user_text = str(state.get("current_transcript", ""))
        pending_draft_raw = state.get("task_pending_draft")

        # ── CONTINUATION: User is responding to a pending confirmation ────────
        if pending_draft_raw:
            import json as _json

            try:
                draft = TaskDraft.model_validate(
                    _json.loads(pending_draft_raw)
                    if isinstance(pending_draft_raw, str)
                    else pending_draft_raw
                )
            except Exception:
                # Corrupt draft — clear and start fresh
                yield Event(
                    author=self.name,
                    actions=EventActions(
                        state_delta={
                            "task_pending_draft": None,
                            "task_pending_confirmation": False,
                        }
                    ),
                )
                return

            confirmed, updated_draft = await self._check_confirmation(user_text, draft)

            if confirmed:
                # User said yes — write to DB and clear state
                task_id = await self._save_task(
                    updated_draft, str(state.get("user_id", ""))
                )
                yield Event(
                    author=self.name,
                    actions=EventActions(
                        state_delta={
                            "task_pending_draft": None,
                            "task_pending_confirmation": False,
                            "task_just_saved_id": task_id,
                            "response_text": "Done! I've saved that for you.",
                        }
                    ),
                )
            else:
                # User corrected — update draft silently and re-confirm
                updated_draft.correction_count += 1
                confirm_text = self._build_confirmation_prompt(updated_draft)
                yield Event(
                    author=self.name,
                    actions=EventActions(
                        state_delta={
                            "task_pending_draft": updated_draft.model_dump_json(),
                            "task_pending_confirmation": True,
                            "response_text": confirm_text,
                        }
                    ),
                )
            return

        # ── FRESH: Extract a new task from user text ──────────────────────────
        draft = await self._extract_task_draft(
            user_text, str(state.get("user_id", ""))
        )
        if draft:
            confirm_text = self._build_confirmation_prompt(draft)
            yield Event(
                author=self.name,
                actions=EventActions(
                    state_delta={
                        "task_pending_draft": draft.model_dump_json(),
                        "task_pending_confirmation": True,
                        "response_text": confirm_text,
                    }
                ),
            )
        else:
            yield Event(
                author=self.name,
                actions=EventActions(
                    state_delta={
                        "response_text": "I couldn't figure out the task details. Could you tell me again?",
                    }
                ),
            )

    async def _extract_task_draft(self, user_text: str, user_id: str) -> TaskDraft | None:
        try:
            result = await generate_json(
                task="task_draft_extraction",
                prompt=f"User said: {user_text}\nUser ID: {user_id}",
                system=TASK_DRAFT_SYSTEM,
            )
            draft = TaskDraft.model_validate(result)
            if not draft.title.strip():
                return None
            return draft
        except Exception as exc:
            logger.error("Task draft extraction failed: %s", exc)
            return None

    async def _check_confirmation(
        self, user_text: str, draft: TaskDraft
    ) -> tuple[bool, TaskDraft]:
        """Return (confirmed, updated_draft). LLM decides if user said yes or corrected."""
        result = await generate_json(
            task="task_confirmation",
            prompt=f"Draft: {draft.model_dump_json()}\nUser said: {user_text}",
            system=(
                'Determine if the user confirmed the task or corrected it. '
                'Return JSON: {"confirmed": bool, "title": str, '
                '"description": str|null, "due_at": str|null, "reminder_at": str|null}'
            ),
        )
        confirmed = bool(result.get("confirmed", False))
        if not confirmed:
            # Apply corrections from LLM
            draft = TaskDraft(
                title=result.get("title", draft.title),
                description=result.get("description", draft.description),
                due_at=result.get("due_at", draft.due_at),
                reminder_at=result.get("reminder_at", draft.reminder_at),
                correction_count=draft.correction_count,
            )
        return confirmed, draft

    def _build_confirmation_prompt(self, draft: TaskDraft) -> str:
        parts = [f"Got it — I'll remind you to {draft.title}"]
        if draft.due_at:
            parts.append(f"due {draft.due_at}")
        if draft.reminder_at:
            parts.append(f"with a reminder at {draft.reminder_at}")
        return f"{', '.join(parts)}. Does that sound right?"

    async def _save_task(self, draft: TaskDraft, user_id: str) -> str:
        client = await get_service_client()
        result = (
            await client.table("tasks")
            .insert(
                {
                    "user_id": user_id,
                    "title": draft.title,
                    "description": draft.description,
                    "due_at": draft.due_at,
                    "reminder_at": draft.reminder_at,
                    "status": "pending",
                }
            )
            .execute()
        )
        return str(result.data[0]["id"]) if result.data else ""


# ── Singleton ─────────────────────────────────────────────────────────────────
task_agent = TaskAgent(name="task_agent")
