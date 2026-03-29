"""Session routes — create and retrieve conversation sessions."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from api.routes._deps import get_current_user_id
from services.supabase_client import get_service_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.post("", status_code=201)
async def create_session(
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict[str, Any]:
    """Create a new conversation session for the authenticated user."""
    try:
        client = await get_service_client()
        result = (
            await client.table("sessions")
            .insert(
                {
                    "user_id": user_id,
                    "status": "active",
                    "started_at": datetime.now(UTC).isoformat(),
                    "turn_count": 0,
                }
            )
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to create session")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("create_session failed [user=%s]: %s", user_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict[str, Any]:
    """Retrieve a session belonging to the authenticated user."""
    try:
        client = await get_service_client()
        result = (
            await client.table("sessions")
            .select("*")
            .eq("id", session_id)
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Session not found")
        return result.data
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("get_session failed [session=%s user=%s]: %s", session_id, user_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error")
