"""BEAN AI v1 — Tasks / Reminders routes."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import AwareDatetime, BaseModel, ConfigDict

from api.routes.auth import get_current_bearer_token, get_current_user_id
from services.calendar_service import CalendarService, get_calendar_token
from services.supabase_client import get_authed_client
from shared.schemas import TaskCreate, TaskResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/tasks", tags=["tasks"])

NO_CACHE_HEADERS = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Pragma": "no-cache",
    "Expires": "0",
}

TaskStatus = Literal["pending", "reminded", "completed", "snoozed", "cancelled"]


class TaskListResponse(BaseModel):
    tasks: list[TaskResponse]
    limit: int
    offset: int
    count: int


class TaskUpdate(BaseModel):
    """Partial task update payload."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    description: str | None = None
    due_at: AwareDatetime | None = None
    reminder_at: AwareDatetime | None = None
    calendar_event_id: str | None = None
    status: TaskStatus | None = None


@router.get("/", response_model=TaskListResponse)
async def list_tasks(
    response: Response,
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
    status_filter: TaskStatus | None = Query(default=None, alias="status"),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
) -> TaskListResponse:
    """List the authenticated user's tasks with pagination."""
    response.headers.update(NO_CACHE_HEADERS)

    try:
        client = await get_authed_client(token)
        query = (
            client.table("tasks")
            .select(
                "id, user_id, title, description, due_at, reminder_at, "
                "calendar_event_id, status, created_at, updated_at",
                count="exact",
            )
            .eq("user_id", current_user_id)
            .order("due_at", desc=False)
            .order("id", desc=True)
            .range(offset, offset + limit - 1)
        )

        if status_filter:
            query = query.eq("status", status_filter)

        result = await query.execute()
        rows = result.data or []
        total_count = result.count or 0

        return TaskListResponse(
            tasks=[TaskResponse.model_validate(row) for row in rows],
            limit=limit,
            offset=offset,
            count=total_count,
        )

    except Exception as exc:
        logger.exception(
            "Failed to list tasks [user=%s status=%s limit=%s offset=%s]: %s",
            current_user_id,
            status_filter,
            limit,
            offset,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list tasks",
        ) from exc


@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    body: TaskCreate,
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
) -> TaskResponse:
    """Create a new task for the authenticated user."""
    try:
        client = await get_authed_client(token)

        create_payload = body.model_dump(exclude_unset=True, mode="json")
        create_payload["user_id"] = current_user_id
        create_payload["status"] = "pending"

        result = (
            await client.table("tasks")
            .insert(create_payload)
            .execute()
        )

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Task creation returned no data",
            )

        return TaskResponse.model_validate(result.data[0])

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to create task [user=%s]: %s", current_user_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create task",
        ) from exc


@router.patch("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: UUID,
    body: TaskUpdate,
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
) -> TaskResponse:
    """Partially update a task owned by the authenticated user."""
    update_payload = body.model_dump(exclude_unset=True, mode="json")

    if not update_payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No fields provided for update",
        )

    update_payload["updated_at"] = datetime.now(UTC).isoformat()

    try:
        client = await get_authed_client(token)
        result = (
            await client.table("tasks")
            .update(update_payload)
            .eq("id", str(task_id))
            .eq("user_id", current_user_id)
            .execute()
        )

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found",
            )

        return TaskResponse.model_validate(result.data[0])

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to update task [user=%s task_id=%s]: %s",
            current_user_id,
            task_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update task",
        ) from exc


@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: UUID,
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
) -> Response:
    """Delete a task owned by the authenticated user.

    If the task has a linked Google Calendar event, that event is also deleted.
    Calendar deletion is best-effort — a Calendar failure never blocks the DB delete.
    """
    try:
        client = await get_authed_client(token)

        # Fetch first to get calendar_event_id before deleting
        fetch_result = (
            await client.table("tasks")
            .select("calendar_event_id")
            .eq("id", str(task_id))
            .eq("user_id", current_user_id)
            .maybe_single()
            .execute()
        )

        if not fetch_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Task not found",
            )

        calendar_event_id = fetch_result.data.get("calendar_event_id")

        # Delete from DB — RLS + user_id check enforce ownership
        await (
            client.table("tasks")
            .delete()
            .eq("id", str(task_id))
            .eq("user_id", current_user_id)
            .execute()
        )

        # Best-effort Calendar cleanup — never fail the request if this errors
        if calendar_event_id:
            try:
                access_token = await get_calendar_token(current_user_id)
                if access_token:
                    cal = CalendarService(access_token)
                    calendar_id = await cal.find_or_create_bean_calendar()
                    await cal.delete_event(calendar_id, calendar_event_id)
            except Exception as cal_exc:
                logger.warning(
                    "Calendar event deletion failed [task=%s event=%s]: %s",
                    task_id,
                    calendar_event_id,
                    cal_exc,
                )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to delete task [user=%s task_id=%s]: %s",
            current_user_id,
            task_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete task",
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)