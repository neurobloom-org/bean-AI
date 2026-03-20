"""BEAN AI v5 — Emotions routes.

Returns aggregated emotion/session data for the mobile app and clinician views.

Design:
- Aggregation happens in Postgres via RPC functions, not in Python memory.
- Auth uses FastAPI dependencies instead of request.state.
- Queries run through a USER-scoped Supabase client so RLS is enforced.
- Timezone offset is supplied by the client so graph buckets match the user's
  local day/week instead of raw UTC boundaries.
"""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.routes.auth import get_current_bearer_token, get_current_user_id
from services.supabase_client import get_authed_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/emotions", tags=["emotions"])


# ─────────────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────────────

class DailyEmotionPoint(BaseModel):
    day: str = Field(description="Local calendar day in YYYY-MM-DD format.")
    emotion: str = Field(description="Emotion label.")
    count: int = Field(description="Number of emotion events recorded that day.")
    avg_confidence: float = Field(description="Average confidence for that emotion/day.")


class DailyEmotionSummaryResponse(BaseModel):
    daily_summary: list[DailyEmotionPoint]
    days: int
    user_id: str


class WeeklyActivityPoint(BaseModel):
    week: str = Field(description="Week start date (local Monday) in YYYY-MM-DD format.")
    session_count: int
    total_duration_seconds: int
    avg_turns_per_session: float
    most_common_emotion: str | None = None


class WeeklyActivityResponse(BaseModel):
    weekly_activity: list[WeeklyActivityPoint]
    weeks: int
    user_id: str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _target_user_id(
    current_user_id: str,
    patient_id: UUID | None,
) -> str:
    """Return the target user id.

    If patient_id is provided, RLS must allow the caller to read that patient's
    rows through the user-scoped Supabase client.
    """
    return str(patient_id) if patient_id else current_user_id


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/summary/daily",
    response_model=DailyEmotionSummaryResponse,
)
async def daily_emotion_summary(
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
    days: int = Query(default=30, ge=1, le=365),
    patient_id: UUID | None = Query(default=None),
    tz_offset_minutes: int = Query(
        default=0,
        ge=-840,
        le=840,
        description="Client timezone offset from UTC in minutes, e.g. -480 for UTC-8.",
    ),
) -> DailyEmotionSummaryResponse:
    """Return daily aggregated emotion counts for the past N days.

    Aggregation happens inside Postgres so the API only returns summary rows.
    """
    user_id = _target_user_id(current_user_id, patient_id)
    client = await get_authed_client(token)

    try:
        result = await client.rpc(
            "get_daily_emotion_summary",
            {
                "p_target_user_id": user_id,
                "p_days": days,
                "p_tz_offset_minutes": tz_offset_minutes,
            },
        ).execute()
    except Exception as exc:
        logger.exception(
            "Failed to fetch daily emotion summary [user_id=%s patient_id=%s days=%s tz_offset=%s]: %s",
            current_user_id,
            patient_id,
            days,
            tz_offset_minutes,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch daily emotion summary",
        ) from exc

    rows = result.data or []
    return DailyEmotionSummaryResponse(
        daily_summary=[DailyEmotionPoint(**row) for row in rows],
        days=days,
        user_id=user_id,
    )


@router.get(
    "/summary/weekly",
    response_model=WeeklyActivityResponse,
)
async def weekly_session_activity(
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
    weeks: int = Query(default=12, ge=1, le=52),
    patient_id: UUID | None = Query(default=None),
    tz_offset_minutes: int = Query(
        default=0,
        ge=-840,
        le=840,
        description="Client timezone offset from UTC in minutes, e.g. -480 for UTC-8.",
    ),
) -> WeeklyActivityResponse:
    """Return weekly aggregated session activity."""
    user_id = _target_user_id(current_user_id, patient_id)
    client = await get_authed_client(token)

    try:
        result = await client.rpc(
            "get_weekly_session_activity",
            {
                "p_target_user_id": user_id,
                "p_weeks": weeks,
                "p_tz_offset_minutes": tz_offset_minutes,
            },
        ).execute()
    except Exception as exc:
        logger.exception(
            "Failed to fetch weekly session activity [user_id=%s patient_id=%s weeks=%s tz_offset=%s]: %s",
            current_user_id,
            patient_id,
            weeks,
            tz_offset_minutes,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch weekly session activity",
        ) from exc

    rows = result.data or []
    return WeeklyActivityResponse(
        weekly_activity=[WeeklyActivityPoint(**row) for row in rows],
        weeks=weeks,
        user_id=user_id,
    )