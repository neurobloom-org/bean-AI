"""Alert routes — list safety alerts for the authenticated user."""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.routes._deps import get_current_user_id
from services.supabase_client import get_service_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


@router.get("")
async def list_alerts(
    user_id: Annotated[str, Depends(get_current_user_id)],
    limit: int = Query(20, ge=1, le=100),
    unacknowledged_only: bool = Query(False),
) -> list[dict[str, Any]]:
    """List safety alerts for the authenticated user, most recent first."""
    try:
        client = await get_service_client()
        query = (
            client.table("alerts")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
        )
        if unacknowledged_only:
            query = query.eq("acknowledged", False)

        result = await query.execute()
        return result.data or []
    except Exception as exc:
        logger.error("list_alerts failed [user=%s]: %s", user_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.patch("/{alert_id}/acknowledge")
async def acknowledge_alert(
    alert_id: str,
    user_id: Annotated[str, Depends(get_current_user_id)],
) -> dict[str, Any]:
    """Mark an alert as acknowledged by the user."""
    try:
        client = await get_service_client()
        result = (
            await client.table("alerts")
            .update({"acknowledged": True})
            .eq("id", alert_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(status_code=404, detail="Alert not found")
        return result.data[0]
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("acknowledge_alert failed [alert=%s user=%s]: %s", alert_id, user_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error")
