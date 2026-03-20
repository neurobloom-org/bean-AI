"""BEAN AI v1 — Sessions routes."""

from __future__ import annotations

import logging
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from api.routes.auth import get_current_bearer_token, get_current_user_id
from services.supabase_client import get_authed_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])

CACHE_CONTROL_HEADER = "private, max-age=15"


# ─────────────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────────────

class SessionSummary(BaseModel):
    id: UUID
    started_at: str
    ended_at: str | None = None
    duration_seconds: int | None = None
    dominant_emotion: str | None = None
    turn_count: int
    status: str


class SessionListResponse(BaseModel):
    sessions: list[SessionSummary]


class SessionDetail(SessionSummary):
    route_distribution: dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/", response_model=SessionListResponse)
async def list_sessions(
    response: Response,
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
    limit: int = Query(default=20, ge=1, le=100),
) -> SessionListResponse:
    """List the authenticated user's sessions.

    Returns metadata only — never transcript/content.
    Queries run through a user-scoped client so RLS is enforced by Postgres.
    """
    response.headers["Cache-Control"] = CACHE_CONTROL_HEADER

    try:
        client = await get_authed_client(token)
        result = (
            await client.table("sessions")
            .select(
                "id, started_at, ended_at, duration_seconds, "
                "dominant_emotion, turn_count, status"
            )
            .eq("user_id", current_user_id)
            .order("started_at", desc=True)
            .limit(limit)
            .execute()
        )

        return SessionListResponse(
            sessions=[
                SessionSummary.model_validate(row)
                for row in (result.data or [])
            ]
        )

    except Exception as exc:
        logger.exception(
            "Failed to list sessions [user=%s limit=%s]: %s",
            current_user_id,
            limit,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list sessions",
        ) from exc


@router.get("/{session_id}", response_model=SessionDetail)
async def get_session(
    response: Response,
    session_id: UUID,
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
) -> SessionDetail:
    """Get metadata for a specific session.

    Uses a user-scoped client so RLS enforces ownership at the database layer.
    """
    response.headers["Cache-Control"] = CACHE_CONTROL_HEADER

    try:
        client = await get_authed_client(token)
        result = (
            await client.table("sessions")
            .select(
                "id, started_at, ended_at, duration_seconds, "
                "dominant_emotion, turn_count, status, route_distribution"
            )
            .eq("id", str(session_id))
            .eq("user_id", current_user_id)  # defense in depth on top of RLS
            .maybe_single()
            .execute()
        )

        if not result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )

        return SessionDetail.model_validate(result.data)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to fetch session [user=%s session_id=%s]: %s",
            current_user_id,
            session_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch session",
        ) from exc