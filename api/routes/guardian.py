"""BEAN AI v1 — Guardian / Doctor dashboard routes.

Privacy constraints:
  ✓ Guardians see: alerts, emotion graphs, session metadata
  ✗ Guardians NEVER see: raw transcripts, conversation content, exact trigger phrases
  ✓ All data is aggregated/summarised before returning
  ✓ Patient must explicitly approve guardian link
"""

from __future__ import annotations

import asyncio
import logging
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

from api.routes.auth import get_current_bearer_token, get_current_user_id
from services.supabase_client import get_authed_client, get_service_client
from shared.schemas import GuardianLinkCreate

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/guardian", tags=["guardian"])


# ─────────────────────────────────────────────────────────────────────────────
# Response models
# ─────────────────────────────────────────────────────────────────────────────

class GuardianLinkResponse(BaseModel):
    id: UUID
    guardian_user_id: UUID
    patient_user_id: UUID
    relationship: str
    can_view_alerts: bool | None = None
    can_view_graphs: bool | None = None
    created_at: str | None = None


class GuardianLinkListResponse(BaseModel):
    links: list[GuardianLinkResponse]


class LinkedPatientSummary(BaseModel):
    patient_user_id: UUID
    display_name: str | None = None
    diagnosis_tags: list[str] = Field(default_factory=list)
    relationship: str | None = None
    can_view_alerts: bool | None = None
    can_view_graphs: bool | None = None
    last_profile_update: str | None = None


class GuardianPatientsResponse(BaseModel):
    patients: list[LinkedPatientSummary]


class PatientProfileSummary(BaseModel):
    display_name: str | None = None
    diagnosis_tags: list[str] = Field(default_factory=list)
    last_profile_update: str | None = None


class EmotionSummaryPoint(BaseModel):
    day: str
    emotion: str
    count: int
    avg_confidence: float


class GuardianAlertSummary(BaseModel):
    id: UUID
    alert_level: str
    alert_factors: list[str] = Field(default_factory=list)
    sms_sent: bool
    acknowledged: bool
    created_at: str


class PatientOverviewResponse(BaseModel):
    patient_user_id: UUID
    relationship: str
    profile: PatientProfileSummary
    emotion_summary_7d: list[EmotionSummaryPoint]
    recent_alerts: list[GuardianAlertSummary]
    sessions_this_week: int
    total_duration_this_week_seconds: int
    avg_turns_per_session_this_week: float
    most_common_emotion_this_week: str | None = None
    last_session_at: str | None = None
    partial_failures: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_tz_offset(value: int) -> int:
    if value < -840 or value > 840:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="tz_offset_minutes must be between -840 and 840",
        )
    return value


def _resolve_result(
    name: str,
    task: Any,
    *,
    fallback: Any,
    partial_failures: list[str],
) -> Any:
    """Resolve an asyncio.gather result with graceful degradation.

    asyncio.gather(return_exceptions=True) returns Exception objects instead
    of raising them. This helper normalises those into a fallback value and
    records the failure for the response's partial_failures field.
    """
    if isinstance(task, Exception):
        logger.exception("Guardian dashboard partial failure [%s]: %s", name, task)
        partial_failures.append(name)
        return fallback
    return task


# ─────────────────────────────────────────────────────────────────────────────
# Guardian link management
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/link", status_code=status.HTTP_201_CREATED)
async def create_guardian_link(
    body: GuardianLinkCreate,
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
) -> dict[str, Any]:
    """Patient creates a link to their guardian/doctor.

    NOTE: GuardianLinkCreate in shared/schemas.py must have guardian_user_id: UUID
    field. The patient IS current_user_id — they provide the guardian's user_id.
    """
    # Verify the guardian account exists before creating the link.
    # Uses service client for admin.get_user_by_id — this is correct,
    # not a privilege escalation since we're only checking existence.
    try:
        admin_client = await get_service_client()
        guardian_auth = await admin_client.auth.admin.get_user_by_id(
            str(body.guardian_user_id)
        )
        if not guardian_auth.user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Guardian user not found. They must register a BEAN account first.",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Guardian account verification failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify guardian account",
        ) from exc

    # Insert via user-scoped client so RLS enforces patient_user_id = auth.uid()
    try:
        client = await get_authed_client(token)
        result = (
            await client.table("guardian_links")
            .insert(
                {
                    "guardian_user_id": str(body.guardian_user_id),
                    "patient_user_id": current_user_id,
                    "relationship": body.relationship,
                    "can_view_alerts": True,
                    "can_view_graphs": True,
                }
            )
            .execute()
        )
        return {
            "message": "Guardian linked successfully.",
            "link": result.data[0] if result.data else {},
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to create guardian link [patient=%s guardian=%s]: %s",
            current_user_id,
            body.guardian_user_id,
            exc,
        )
        if "unique" in str(exc).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This guardian link already exists.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create guardian link",
        ) from exc


@router.get("/links", response_model=GuardianLinkListResponse)
async def list_my_guardian_links(
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
) -> GuardianLinkListResponse:
    """List all guardian links where the caller is either the patient or guardian."""
    try:
        client = await get_authed_client(token)
        result = (
            await client.table("guardian_links")
            .select(
                "id, guardian_user_id, patient_user_id, relationship, "
                "can_view_alerts, can_view_graphs, created_at"
            )
            .or_(
                f"guardian_user_id.eq.{current_user_id},"
                f"patient_user_id.eq.{current_user_id}"
            )
            .execute()
        )
        return GuardianLinkListResponse(
            links=[GuardianLinkResponse(**row) for row in (result.data or [])]
        )
    except Exception as exc:
        logger.exception(
            "Failed to list guardian links [user=%s]: %s", current_user_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list guardian links",
        ) from exc


@router.delete("/link/{link_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_guardian_link(
    link_id: UUID,
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
) -> Response:
    """Either the patient or the guardian may remove a link."""
    try:
        client = await get_authed_client(token)

        # Fetch first to verify caller is a party to this link
        link_result = (
            await client.table("guardian_links")
            .select("id, guardian_user_id, patient_user_id")
            .eq("id", str(link_id))
            .maybe_single()
            .execute()
        )

        link = link_result.data
        if not link:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Link not found or you do not have permission to remove it.",
            )

        if current_user_id not in {link["guardian_user_id"], link["patient_user_id"]}:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to remove this link.",
            )

        delete_result = (
            await client.table("guardian_links")
            .delete()
            .eq("id", str(link_id))
            .execute()
        )

        if not delete_result.data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Link not found or already removed.",
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to remove guardian link [link_id=%s user=%s]: %s",
            link_id,
            current_user_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to remove guardian link",
        ) from exc

    return Response(status_code=status.HTTP_204_NO_CONTENT)


# ─────────────────────────────────────────────────────────────────────────────
# Guardian dashboard — patient list
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/patients", response_model=GuardianPatientsResponse)
async def list_my_patients(
    response: Response,
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
) -> GuardianPatientsResponse:
    """List all patients linked to the authenticated guardian.

    FIX: Replaced get_guardian_patients RPC call — that function does not
    exist in our migrations (004_functions.sql). Using two table queries instead:
    1. guardian_links to get patient_ids + permission flags
    2. user_profiles to get display names + diagnosis_tags
    RLS on both tables enforces the guardian can only see their own links.
    """
    response.headers["Cache-Control"] = "private, max-age=15"

    try:
        client = await get_authed_client(token)

        links_result = (
            await client.table("guardian_links")
            .select(
                "patient_user_id, relationship, can_view_alerts, can_view_graphs"
            )
            .eq("guardian_user_id", current_user_id)
            .execute()
        )

        links = links_result.data or []
        if not links:
            return GuardianPatientsResponse(patients=[])

        patient_ids = [link["patient_user_id"] for link in links]
        link_map = {link["patient_user_id"]: link for link in links}

        profiles_result = (
            await client.table("user_profiles")
            .select("user_id, display_name, diagnosis_tags, updated_at")
            .in_("user_id", patient_ids)
            .execute()
        )

        profile_map = {
            p["user_id"]: p for p in (profiles_result.data or [])
        }

        patients = []
        for patient_id in patient_ids:
            link = link_map[patient_id]
            profile = profile_map.get(patient_id, {})
            patients.append(
                LinkedPatientSummary(
                    patient_user_id=patient_id,
                    display_name=profile.get("display_name"),
                    diagnosis_tags=profile.get("diagnosis_tags", []),
                    relationship=link.get("relationship"),
                    can_view_alerts=link.get("can_view_alerts"),
                    can_view_graphs=link.get("can_view_graphs"),
                    last_profile_update=profile.get("updated_at"),
                )
            )

        return GuardianPatientsResponse(patients=patients)

    except Exception as exc:
        logger.exception(
            "Failed to list patients [guardian=%s]: %s", current_user_id, exc
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list patients",
        ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# Guardian dashboard — patient overview
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/patients/{patient_id}/overview", response_model=PatientOverviewResponse)
async def get_patient_overview(
    patient_id: UUID,
    current_user_id: Annotated[str, Depends(get_current_user_id)],
    token: Annotated[str, Depends(get_current_bearer_token)],
    tz_offset_minutes: int = Query(default=0),
) -> PatientOverviewResponse:
    """Get a privacy-safe overview for a specific patient.

    Fetches profile, emotions, alerts, session activity, and last session
    concurrently using asyncio.gather. Any individual failure degrades
    gracefully — partial_failures in the response lists what failed.

    NEVER returns: raw transcripts, conversation text, trigger phrases.
    """
    tz_offset_minutes = _validate_tz_offset(tz_offset_minutes)
    patient_id_str = str(patient_id)

    try:
        client = await get_authed_client(token)

        # Step 1 — verify the guardian link exists and get permission flags.
        # Must happen before the parallel fetches so we can gate optional queries.
        link_result = (
            await client.table("guardian_links")
            .select("relationship, can_view_alerts, can_view_graphs")
            .eq("guardian_user_id", current_user_id)
            .eq("patient_user_id", patient_id_str)
            .maybe_single()
            .execute()
        )

        if not link_result.data:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You are not linked to this patient.",
            )

        link = link_result.data
        partial_failures: list[str] = []

        # Step 2 — fire parallel queries with graceful degradation.
        async def _safe(coro):
            """Wrap optional coroutines — None means the caller opted out."""
            return await coro if coro is not None else None

        (
            profile_result,
            emotion_result,
            alerts_result,
            weekly_result,
            last_session_result,
        ) = await asyncio.gather(
            client.table("user_profiles")
            .select("display_name, diagnosis_tags, updated_at")
            .eq("user_id", patient_id_str)
            .maybe_single()
            .execute(),

            _safe(
                client.rpc(
                    "get_daily_emotion_summary",
                    {
                        "p_target_user_id": patient_id_str,
                        "p_days": 7,
                        "p_tz_offset_minutes": tz_offset_minutes,
                    },
                ).execute()
                if link.get("can_view_graphs")
                else None
            ),

            _safe(
                client.table("alerts")
                .select(
                    "id, alert_level, alert_factors, sms_sent, acknowledged, created_at"
                )
                .eq("user_id", patient_id_str)
                .eq("acknowledged", False)
                .order("created_at", desc=True)
                .limit(10)
                .execute()
                if link.get("can_view_alerts")
                else None
            ),

            client.rpc(
                "get_weekly_session_activity",
                {
                    "p_target_user_id": patient_id_str,
                    "p_weeks": 1,
                    "p_tz_offset_minutes": tz_offset_minutes,
                },
            ).execute(),

            client.table("sessions")
            .select("started_at")
            .eq("user_id", patient_id_str)
            .order("started_at", desc=True)
            .limit(1)
            .execute(),

            return_exceptions=True,
        )

        # Step 3 — resolve results, recording any partial failures
        profile_result = _resolve_result(
            "profile", profile_result, fallback=None, partial_failures=partial_failures
        )
        emotion_result = _resolve_result(
            "emotions", emotion_result, fallback=None, partial_failures=partial_failures
        )
        alerts_result = _resolve_result(
            "alerts", alerts_result, fallback=None, partial_failures=partial_failures
        )
        weekly_result = _resolve_result(
            "weekly_activity", weekly_result, fallback=None, partial_failures=partial_failures
        )
        last_session_result = _resolve_result(
            "last_session", last_session_result, fallback=None, partial_failures=partial_failures
        )

        profile = (profile_result.data if profile_result else None) or {}
        emotion_rows = (emotion_result.data if emotion_result else None) or []
        alert_rows = (alerts_result.data if alerts_result else None) or []
        weekly_rows = (weekly_result.data if weekly_result else None) or []
        current_week = weekly_rows[-1] if weekly_rows else None
        last_session_at = (
            last_session_result.data[0]["started_at"]
            if last_session_result and last_session_result.data
            else None
        )

        return PatientOverviewResponse(
            patient_user_id=patient_id,
            relationship=link["relationship"],
            profile=PatientProfileSummary(
                display_name=profile.get("display_name"),
                diagnosis_tags=profile.get("diagnosis_tags", []),
                last_profile_update=profile.get("updated_at"),
            ),
            emotion_summary_7d=[EmotionSummaryPoint(**row) for row in emotion_rows],
            recent_alerts=[GuardianAlertSummary(**row) for row in alert_rows],
            sessions_this_week=int(current_week["session_count"]) if current_week else 0,
            total_duration_this_week_seconds=(
                int(current_week["total_duration_seconds"]) if current_week else 0
            ),
            avg_turns_per_session_this_week=(
                float(current_week["avg_turns_per_session"]) if current_week else 0.0
            ),
            most_common_emotion_this_week=(
                current_week.get("most_common_emotion") if current_week else None
            ),
            last_session_at=last_session_at,
            partial_failures=partial_failures,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to fetch patient overview [guardian=%s patient=%s]: %s",
            current_user_id,
            patient_id,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch patient overview",
        ) from exc