"""Shared FastAPI dependencies for all routes."""

from __future__ import annotations

from fastapi import Header, HTTPException, Request

from shared.config import get_settings


async def get_current_user_id(request: Request) -> str:
    """Extract the authenticated user_id set by SupabaseAuthMiddleware."""
    user_id: str | None = getattr(request.state, "user_id", None)
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user_id


async def verify_internal_key(x_internal_key: str = Header(...)) -> None:
    """Verify the X-Internal-Key header for /internal/* routes."""
    settings = get_settings()
    if not settings.internal_api_key or x_internal_key != settings.internal_api_key:
        raise HTTPException(status_code=403, detail="Forbidden")
