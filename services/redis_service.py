"""BEAN AI — Redis client singleton.

Used for:
  - 10-turn conversation cache per session (hot path)
  - Task confirmation state per session

Key patterns:
  convo:{session_id}          → JSON list of last 10 turns
  task_draft:{session_id}     → JSON of pending task draft awaiting confirmation
"""

from __future__ import annotations

import json
import logging
from typing import Any

import redis.asyncio as aioredis

from shared.config import get_settings

logger = logging.getLogger(__name__)

_redis: aioredis.Redis | None = None  # type: ignore[type-arg]


async def get_redis() -> aioredis.Redis:  # type: ignore[type-arg]
    """Return the shared Redis client, connecting on first call."""
    global _redis
    if _redis is None:
        settings = get_settings()
        _redis = aioredis.from_url(
            settings.redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
        )
    return _redis


async def redis_get(key: str) -> Any | None:
    r = await get_redis()
    raw = await r.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw


async def redis_set(key: str, value: Any, ttl_seconds: int = 86400) -> None:
    r = await get_redis()
    await r.set(key, json.dumps(value, ensure_ascii=False), ex=ttl_seconds)


async def redis_delete(key: str) -> None:
    r = await get_redis()
    await r.delete(key)