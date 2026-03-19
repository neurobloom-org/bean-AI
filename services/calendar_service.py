"""
services/calendar_service.py
=============================
Calendar service for managing reminders and tasks.
Needs supabase_client.
"""

from datetime import datetime, timezone

from services.supabase_client import supabase


async def create_reminder(
    user_id: str,
    title: str,
    remind_at: datetime,
    notes: str = "",
) -> dict:
    """Create a new reminder for a user."""
    result = (
        supabase.table("tasks")
        .insert(
            {
                "user_id": user_id,
                "title": title,
                "remind_at": remind_at.isoformat(),
                "notes": notes,
                "completed": False,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        .execute()
    )
    return result.data[0]


async def get_upcoming_reminders(user_id: str) -> list[dict]:
    """Get all upcoming reminders for a user."""
    now = datetime.now(timezone.utc).isoformat()
    result = (
        supabase.table("tasks")
        .select("*")
        .eq("user_id", user_id)
        .eq("completed", False)
        .gte("remind_at", now)
        .order("remind_at")
        .execute()
    )
    return result.data


async def complete_reminder(task_id: str) -> bool:
    """Mark a reminder as completed."""
    supabase.table("tasks").update({"completed": True}).eq("id", task_id).execute()
    return True


async def delete_reminder(task_id: str) -> bool:
    """Delete a reminder."""
    supabase.table("tasks").delete().eq("id", task_id).execute()
    return True
