"""BEAN AI — Tiered LLM service.

Design principle: minimise cost without compromising safety-critical responses.

CHEAP tier (Gemini Flash):
  - Routing decisions        (~$0.00001/call)
  - Casual conversation      (~$0.0001/turn)
  - Fact extraction          (~$0.00005/turn)
  - Memory similarity query  (~$0.00005/call)
  - Task / reminder parsing  (~$0.00003/call)
  - Music selection          (~$0.00002/call)

PRO tier (Gemini Pro):
  - Therapeutic conversation  (~$0.005/turn)
  - Safety / alert analysis   (~$0.003/call)
  - Active listening          (~$0.005/turn)
  - Crisis response           (~$0.005/turn)

Estimated saving vs "all Pro": ~85% reduction in LLM costs.

SDK note:
  Uses google-genai (google.genai) — the current supported package.
  The old google-generativeai package is deprecated and no longer receives
  updates. See: https://github.com/google-gemini/deprecated-generative-ai-python
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

from google import genai
from google.genai import types as genai_types

from shared.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task → Tier mapping
# ---------------------------------------------------------------------------


class LLMTier(str, Enum):
    CHEAP = "cheap"
    PRO = "pro"


TASK_TIER_MAP: dict[str, LLMTier] = {
    # ── Cheap tier ────────────────────────────────────────────────────────────
    "routing": LLMTier.CHEAP,
    "casual_chat": LLMTier.CHEAP,
    "fact_extraction": LLMTier.CHEAP,
    "memory_query": LLMTier.CHEAP,
    "task_management": LLMTier.CHEAP,
    "music_selection": LLMTier.CHEAP,
    "emotion_summary": LLMTier.CHEAP,
    # ── Pro tier ──────────────────────────────────────────────────────────────
    "therapeutic_chat": LLMTier.PRO,
    "safety_analysis": LLMTier.PRO,
    "alert_assessment": LLMTier.PRO,
    "active_listening": LLMTier.PRO,
    "crisis_response": LLMTier.PRO,
}


def get_model_for_task(task: str) -> str:
    """Return the model name for a given task based on its cost tier."""
    settings = get_settings()
    tier = TASK_TIER_MAP.get(task, LLMTier.CHEAP)
    model = settings.llm_pro_model if tier == LLMTier.PRO else settings.llm_cheap_model
    logger.debug("Task '%s' → tier=%s model=%s", task, tier.value, model)
    return model


def get_tier(task: str) -> LLMTier:
    """Return the tier for a given task."""
    return TASK_TIER_MAP.get(task, LLMTier.CHEAP)


def _get_client() -> genai.Client:
    """Create a google.genai Client with the configured API key.

    A new Client is created per-call. This is intentional — Client objects
    are lightweight and stateless, and creating one per-call avoids any
    thread-safety concerns with a shared singleton in an async context.
    """
    settings = get_settings()
    return genai.Client(api_key=settings.google_api_key)


# ---------------------------------------------------------------------------
# Core generation functions
# ---------------------------------------------------------------------------


async def generate(
    task: str,
    prompt: str,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Generate a plain text response using the appropriate model for the task.

    Args:
        task:        Task name from TASK_TIER_MAP — determines model tier.
        prompt:      The user/content prompt to send.
        system:      Optional system instruction.
        temperature: Override temperature (defaults to tier-specific value).
        max_tokens:  Override max output tokens (defaults to tier-specific value).

    Returns:
        Generated text response (stripped).

    Raises:
        RuntimeError: If generation fails.
    """
    settings = get_settings()
    model_name = get_model_for_task(task)
    tier = get_tier(task)

    _temperature = (
        temperature
        if temperature is not None
        else (
            settings.llm_pro_temperature
            if tier == LLMTier.PRO
            else settings.llm_cheap_temperature
        )
    )
    _max_tokens = (
        max_tokens
        if max_tokens is not None
        else (
            settings.llm_pro_max_tokens
            if tier == LLMTier.PRO
            else settings.llm_cheap_max_tokens
        )
    )

    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        temperature=_temperature,
        max_output_tokens=_max_tokens,
    )

    try:
        client = _get_client()

        # The new SDK's async interface lives on client.aio — no asyncio.to_thread needed.
        response = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        text = (response.text or "").strip()

        logger.debug(
            "LLM [task=%s tier=%s model=%s] → %d chars",
            task,
            tier.value,
            model_name,
            len(text),
        )
        return text

    except Exception as exc:
        logger.error(
            "LLM generation failed [task=%s model=%s]: %s",
            task,
            model_name,
            exc,
        )
        raise RuntimeError(f"LLM failed for task '{task}': {exc}") from exc


async def generate_json(
    task: str,
    prompt: str,
    system: str | None = None,
) -> dict[str, Any]:
    """Generate and parse a JSON response.

    Forces temperature=0 and strips markdown fences before parsing.

    Raises:
        ValueError: If the model returns non-JSON output.
        RuntimeError: If generation itself fails.
    """
    raw = await generate(
        task=task,
        prompt=prompt,
        system=system,
        temperature=0.0,
        max_tokens=512,
    )

    # Strip markdown code fences (```json ... ```)
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        return dict(json.loads(raw))
    except json.JSONDecodeError as exc:
        logger.error(
            "JSON parse failed [task=%s]: %s\nRaw output (first 300 chars): %s",
            task,
            exc,
            raw[:300],
        )
        raise ValueError(f"LLM returned non-JSON for task '{task}'") from exc


async def generate_stream(
    task: str,
    prompt: str,
    system: str | None = None,
) -> AsyncGenerator[str, None]:
    """Async generator that streams response chunks.

    Usage:
        async for chunk in generate_stream("therapeutic_chat", prompt):
            await ws.send_text(chunk)
    """
    settings = get_settings()
    model_name = get_model_for_task(task)
    tier = get_tier(task)

    _temperature = (
        settings.llm_pro_temperature
        if tier == LLMTier.PRO
        else settings.llm_cheap_temperature
    )

    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        temperature=_temperature,
    )

    try:
        client = _get_client()

        async for chunk in await client.aio.models.generate_content_stream(
            model=model_name,
            contents=prompt,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

    except Exception as exc:
        logger.error("Stream generation failed [task=%s]: %s", task, exc)
        raise


# ---------------------------------------------------------------------------
# Convenience wrappers used by specific agents
# ---------------------------------------------------------------------------


async def route_message(context: str) -> dict[str, Any]:
    """Routing decision — always uses cheap model."""
    return await generate_json("routing", context)


async def extract_facts(prompt: str) -> dict[str, Any]:
    """Memory fact extraction — always uses cheap model."""
    return await generate_json("fact_extraction", prompt)


async def therapeutic_response(prompt: str, system: str) -> str:
    """Therapeutic conversation — always uses Pro model."""
    return await generate("therapeutic_chat", prompt, system=system)


async def assess_safety(prompt: str, system: str) -> dict[str, Any]:
    """Safety / alert assessment — always uses Pro model."""
    return await generate_json("safety_analysis", prompt, system=system)
