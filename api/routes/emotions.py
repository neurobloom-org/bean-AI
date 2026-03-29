"""Emotion routes — retrieve emotion history for the authenticated user."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.routes._deps import get_current_user_id
from services.supabase_client import get_service_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/emotions", tags=["emotions"])


@router.get("")
async def list_emotions(
    user_id: Annotated[str, Depends(get_current_user_id)],
    days: int = Query(7, ge=1, le=90, description="Number of days of history to return"),
    limit: int = Query(100, ge=1, le=500),
) -> list[dict[str, Any]]:
    """Return recent emotion events for the authenticated user."""
    try:
        client = await get_service_client()
        since = (datetime.now(UTC) - timedelta(days=days)).isoformat()

        result = (
            await client.table("emotion_events")
            .select("id, emotion, confidence, session_id, created_at")
            .eq("user_id", user_id)
            .gt("created_at", since)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as exc:
        logger.error("list_emotions failed [user=%s]: %s", user_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/summary")
async def emotion_summary(
    user_id: Annotated[str, Depends(get_current_user_id)],
    days: int = Query(7, ge=1, le=90),
) -> dict[str, Any]:
    """Return aggregated daily emotion summary via Supabase RPC."""
    try:
        client = await get_service_client()
        result = await client.rpc(
            "get_patient_emotion_summary",
            {"patient_uuid": user_id, "days_back": days},
        ).execute()
        return {"days": days, "summary": result.data or []}
    except Exception as exc:
        logger.error("emotion_summary failed [user=%s]: %s", user_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error")
