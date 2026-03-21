"""BEAN AI v1 — Alerts routes.

Important:
- These routes must use a USER-SCOPED Supabase client, not the service role.
- Using the service role here would bypass RLS and create an IDOR vulnerability.
- Visibility and update permissions are enforced by Supabase RLS.
"""

from __future__ import annotations

import logging
from typing import Any, cast
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Request, status

from services.supabase_client import get_authed_client
from shared.schemas import utcnow

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])

DEFAULT_ALERT_LIMIT = 50
MAX_ALERT_LIMIT = 100


def _extract_bearer_token_from_request(request: Request) -> str:
    """Extract the caller JWT from the Authorization header."""
    auth_header = str(request.headers.get("Authorization", ""))
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required. Use: Authorization: Bearer <token>",
        )

    token = auth_header[7:].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token is missing",
        )

    return str(token)


@router.get("/")  # type: ignore[untyped-decorator]
async def list_alerts(
    request: Request,
    limit: int = Query(
        default=DEFAULT_ALERT_LIMIT,
        ge=1,
        le=MAX_ALERT_LIMIT,
        description="Maximum number of alerts to return.",
    ),
    offset: int = Query(
        default=0,
        ge=0,
        description="Number of alerts to skip for pagination.",
    ),
    unacknowledged_only: bool = Query(
        default=False,
        description="Return only alerts that have not been acknowledged yet.",
    ),
) -> dict[str, Any]:
    """List alerts visible to the authenticated caller."""
    token = _extract_bearer_token_from_request(request)
    client = await get_authed_client(token)

    query = (
        client.table("alerts")
        .select(
            "id, user_id, alert_level, alert_factors, notified_guardian, "
            "sms_sent, acknowledged, acknowledged_at, created_at"
        )
        .order("created_at", desc=True)
        .range(offset, offset + limit - 1)
    )

    if unacknowledged_only:
        query = query.eq("acknowledged", False)

    try:
        result = await query.execute()
    except Exception as exc:
        logger.exception("Failed to list alerts via RLS: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch alerts",
        ) from exc

    # Tell MyPy this is a list of dictionaries
    alerts = cast(list[dict[str, Any]], result.data) if result.data else []

    return {
        "alerts": alerts,
        "limit": limit,
        "offset": offset,
        "count": len(alerts),
    }


@router.patch("/{alert_id}/acknowledge")  # type: ignore[untyped-decorator]
async def acknowledge_alert(alert_id: UUID, request: Request) -> dict[str, bool]:
    """Mark an alert as acknowledged."""
    token = _extract_bearer_token_from_request(request)
    client = await get_authed_client(token)

    try:
        result = (
            await client.table("alerts")
            .update(
                {
                    "acknowledged": True,
                    "acknowledged_at": utcnow().isoformat(),
                }
            )
            .eq("id", str(alert_id))
            .eq("acknowledged", False)
            .execute()
        )
    except Exception as exc:
        logger.exception(
            "Failed to acknowledge alert=%s via RLS: %s",
            alert_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acknowledge alert",
        ) from exc

    if not result.data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found, already acknowledged, or access denied",
        )

    return {"acknowledged": True}
