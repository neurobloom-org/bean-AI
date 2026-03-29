"""Guardian routes — oversight dashboard for linked patients."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query

from api.routes._deps import get_current_user_id
from services.supabase_client import get_service_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/guardian", tags=["guardian"])


@router.get("/patients")
async def list_patients(
    guardian_id: Annotated[str, Depends(get_current_user_id)],
) -> list[dict[str, Any]]:
    """List all patients linked to this guardian account."""
    try:
        client = await get_service_client()
        links = (
            await client.table("guardian_links")
            .select("patient_user_id, relationship, created_at")
            .eq("guardian_user_id", guardian_id)
            .execute()
        )
        if not links.data:
            return []

        patient_ids = [row["patient_user_id"] for row in links.data]
        profiles = (
            await client.table("user_profiles")
            .select("user_id, display_name, preferred_name, diagnosis_tags")
            .in_("user_id", patient_ids)
            .execute()
        )

        profile_map: dict[str, dict[str, Any]] = {
            p["user_id"]: p for p in (profiles.data or [])
        }

        result = []
        for link in links.data:
            pid = link["patient_user_id"]
            profile = profile_map.get(pid, {})
            result.append(
                {
                    "patient_user_id": pid,
                    "relationship": link["relationship"],
                    "linked_at": link["created_at"],
                    "display_name": profile.get("display_name") or profile.get("preferred_name"),
                    "diagnosis_tags": profile.get("diagnosis_tags", []),
                }
            )
        return result
    except Exception as exc:
        logger.error("list_patients failed [guardian=%s]: %s", guardian_id, exc)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/patients/{patient_id}/emotions")
async def patient_emotion_trends(
    patient_id: str,
    guardian_id: Annotated[str, Depends(get_current_user_id)],
    days: int = Query(7, ge=1, le=90),
) -> dict[str, Any]:
    """Return aggregated emotion trends for a linked patient.

    Guardians can only access aggregated emotion labels and counts —
    not raw transcripts or conversation content (enforced by RLS).
    """
    try:
        client = await get_service_client()

        # Verify the guardian-patient relationship exists
        link = (
            await client.table("guardian_links")
            .select("patient_user_id")
            .eq("guardian_user_id", guardian_id)
            .eq("patient_user_id", patient_id)
            .maybe_single()
            .execute()
        )
        if not link.data:
            raise HTTPException(status_code=404, detail="Patient not found or not linked")

        summary = await client.rpc(
            "get_patient_emotion_summary",
            {"patient_uuid": patient_id, "days_back": days},
        ).execute()

        # Recent alerts — level + timestamp only, no conversation content
        since = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        alerts = (
            await client.table("alerts")
            .select("id, alert_level, alert_factors, created_at, acknowledged")
            .eq("user_id", patient_id)
            .gt("created_at", since)
            .order("created_at", desc=True)
            .limit(10)
            .execute()
        )

        return {
            "patient_user_id": patient_id,
            "days": days,
            "emotion_summary": summary.data or [],
            "recent_alerts": alerts.data or [],
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "patient_emotion_trends failed [guardian=%s patient=%s]: %s",
            guardian_id, patient_id, exc,
        )
        raise HTTPException(status_code=500, detail="Internal server error")
