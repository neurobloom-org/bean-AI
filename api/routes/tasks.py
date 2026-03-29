"""Task/reminder routes — create and list tasks for the authenticated user."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.routes._deps import get_current_user_id
from services.supabase_client import get_service_client
from shared.schemas import TaskCreate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])


@router.post("", status_code=201)
async def create_task(
    body: TaskCreate,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict[str, Any]:
    """Create a new reminder task for the authenticated user."""
    try:
        client = await get_service_client()
        payload: dict[str, Any] = {
            "user_id": user_id,
            "title": body.title,
            "status": "pending",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        if body.description is not None:
            payload["description"] = body.description
        if body.due_at is not None:
            payload["due_at"] = body.due_at.isoformat()

        result = await client.table("tasks").insert(payload).execute()
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create task")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("create_task failed [user=%s]: %s", user_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("")
async def list_tasks(
    user_id: Annotated[str, Depends(get_current_user_id)],
    status: str | None = Query(None, description="Filter by status: pending, reminded, completed, snoozed, cancelled"),
    limit: int = Query(50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """List tasks for the authenticated user, optionally filtered by status."""
    try:
        client = await get_service_client()
        query = (
            client.table("tasks")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
        )
        if status:
            query = query.eq("status", status)

        result = await query.execute()
        return result.data or []
    except Exception as exc:
        logger.error("list_tasks failed [user=%s]: %s", user_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error")
