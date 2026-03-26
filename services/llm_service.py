"""BEAN AI — Tiered LLM service.

Design principle: minimise cost without compromising safety-critical responses.

CHEAP tier (Gemini Flash):
  - Routing decisions        (~$0.00001/call)
  - Casual conversation      (~$0.0001/turn)
  - Fact extraction          (~$0.00005/turn)
  - Memory similarity query  (~$0.00005/call)
  - Memory compression       (~$0.00003/call)
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

Exception contract:
  generate()        raises LLMError on generation failure.
  generate_json()   raises LLMError on generation failure, ValueError on bad JSON.
  generate_stream() raises LLMError on generation failure.
  All three propagate asyncio.CancelledError without wrapping.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any

from google import genai
from google.genai import types as genai_types

from shared.config import get_settings
from shared.exceptions import LLMError

__all__ = [
    "generate",
    "generate_json",
    "generate_stream",
    "get_model_for_task",
    "get_tier",
    "route_message",
    "extract_facts",
    "therapeutic_response",
    "assess_safety",
    "LLMTier",
    "TASK_TIER_MAP",
]

logger = logging.getLogger(__name__)

_genai_client: genai.Client | None = None


# ── Task → tier mapping ───────────────────────────────────────────────────────


class LLMTier(str, Enum):
    CHEAP = "cheap"
    PRO = "pro"


TASK_TIER_MAP: dict[str, LLMTier] = {
    # ── Cheap tier ────────────────────────────────────────────────────────────
    "routing":               LLMTier.CHEAP,
    "casual_chat":           LLMTier.CHEAP,
    "fact_extraction":       LLMTier.CHEAP,
    "memory_query":          LLMTier.CHEAP,
    "memory_compress":       LLMTier.CHEAP,   # FIX Bug 5: added — used by MemoryWriterAgent
    "task_management":       LLMTier.CHEAP,
    "task_draft_extraction": LLMTier.CHEAP,
    "task_confirmation":     LLMTier.CHEAP,
    "music_selection":       LLMTier.CHEAP,
    "emotion_summary":       LLMTier.CHEAP,
    # ── Pro tier ──────────────────────────────────────────────────────────────
    "therapeutic_chat":  LLMTier.PRO,
    "safety_analysis":   LLMTier.PRO,
    "alert_assessment":  LLMTier.PRO,
    "active_listening":  LLMTier.PRO,
    "crisis_response":   LLMTier.PRO,
}


def get_model_for_task(task: str) -> str:
    """Return the model name for a given task based on its cost tier."""
    settings = get_settings()
    tier = TASK_TIER_MAP.get(task, LLMTier.CHEAP)
    model = settings.llm_pro_model if tier == LLMTier.PRO else settings.llm_cheap_model
    logger.debug("Task %r → tier=%s model=%s", task, tier.value, model)
    return model


def get_tier(task: str) -> LLMTier:
    """Return the LLMTier for a given task (defaults to CHEAP for unknown tasks)."""
    return TASK_TIER_MAP.get(task, LLMTier.CHEAP)


def _get_client() -> genai.Client:
    """Return a cached google.genai Client configured with the API key.

    Raises:
        LLMError: If GOOGLE_API_KEY is not configured.
    """
    global _genai_client

    if _genai_client is None:
        settings = get_settings()
        if not settings.google_api_key:
            raise LLMError("GOOGLE_API_KEY is not configured — cannot make LLM calls")
        _genai_client = genai.Client(api_key=settings.google_api_key)

    return _genai_client


# ── Core generation functions ─────────────────────────────────────────────────


async def generate(
    task: str,
    prompt: str,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Generate a plain text response using the appropriate model for the task.

    Args:
        task:        Task name from TASK_TIER_MAP — determines model and tier.
        prompt:      The user/content prompt to send.
        system:      Optional system instruction (sent as system_instruction,
                     not as part of the user turn).
        temperature: Override temperature. Defaults to tier-specific setting.
        max_tokens:  Override max output tokens. Defaults to tier-specific setting.

    Returns:
        Generated text response, stripped of leading/trailing whitespace.
        Returns an empty string if the model returns no text (not an error).

    Raises:
        LLMError:              If the API call fails for any reason.
        asyncio.CancelledError: Propagated without wrapping.
    """
    settings = get_settings()
    model_name = get_model_for_task(task)
    tier = get_tier(task)

    resolved_temperature = (
        temperature
        if temperature is not None
        else (
            settings.llm_pro_temperature
            if tier == LLMTier.PRO
            else settings.llm_cheap_temperature
        )
    )
    resolved_max_tokens = (
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
        temperature=resolved_temperature,
        max_output_tokens=resolved_max_tokens,
    )

    try:
        client = _get_client()

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

    except LLMError:
        raise
    except Exception as exc:
        logger.error(
            "LLM generation failed [task=%s model=%s]: %s",
            task,
            model_name,
            exc,
        )
        raise LLMError(f"LLM generation failed for task '{task}': {exc}") from exc


async def generate_json(
    task: str,
    prompt: str,
    system: str | None = None,
) -> dict[str, Any]:
    """Generate and parse a JSON response.

    Forces temperature=0 for deterministic JSON output.
    Strips markdown code fences (```json ... ```) before parsing.

    Args:
        task:   Task name — determines model tier.
        prompt: The user/content prompt. Should instruct the model to return JSON.
        system: Optional system instruction.

    Returns:
        Parsed JSON response as a dict.

    Raises:
        LLMError:  If generation fails.
        ValueError: If the model returns non-JSON output.
        asyncio.CancelledError: Propagated without wrapping.
    """
    raw = await generate(
        task=task,
        prompt=prompt,
        system=system,
        temperature=0.0,
        max_tokens=512,
    )

    # Strip markdown code fences the model may wrap the JSON in.
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()

    try:
        return dict(json.loads(cleaned))
    except json.JSONDecodeError as exc:
        logger.error(
            "JSON parse failed [task=%s]: %s\nRaw output (first 300 chars): %s",
            task,
            exc,
            raw[:300],
        )
        raise ValueError(f"LLM returned non-JSON for task '{task}': {exc}") from exc


async def generate_stream(
    task: str,
    prompt: str,
    system: str | None = None,
) -> AsyncGenerator[str, None]:
    """Async generator that yields response text chunks as they arrive.

    Usage:
        async for chunk in generate_stream("therapeutic_chat", prompt):
            await ws.send_text(chunk)

    Args:
        task:   Task name — determines model tier.
        prompt: The user/content prompt.
        system: Optional system instruction.

    Yields:
        Non-empty text chunks as strings.

    Raises:
        LLMError:              If the stream fails to start or errors mid-stream.
        asyncio.CancelledError: Propagated without wrapping.
    """
    settings = get_settings()
    model_name = get_model_for_task(task)
    tier = get_tier(task)

    resolved_temperature = (
        settings.llm_pro_temperature
        if tier == LLMTier.PRO
        else settings.llm_cheap_temperature
    )

    config = genai_types.GenerateContentConfig(
        system_instruction=system,
        temperature=resolved_temperature,
    )

    try:
        client = _get_client()

        async for chunk in client.aio.models.generate_content_stream(
            model=model_name,
            contents=prompt,
            config=config,
        ):
            if chunk.text:
                yield chunk.text

    except LLMError:
        raise
    except Exception as exc:
        logger.error(
            "LLM stream failed [task=%s model=%s]: %s",
            task,
            model_name,
            exc,
        )
        raise LLMError(f"LLM stream failed for task '{task}': {exc}") from exc


# ── Convenience wrappers ──────────────────────────────────────────────────────


async def route_message(context: str) -> dict[str, Any]:
    """Routing decision — always uses cheap model."""
    return await generate_json("routing", context)


async def extract_facts(prompt: str) -> dict[str, Any]:
    """Memory fact extraction — always uses cheap model."""
    return await generate_json("fact_extraction", prompt)


async def therapeutic_response(prompt: str, system: str) -> str:
    """Therapeutic conversation — always uses Pro model.

    Args:
        prompt: User message content only (not the full system prompt).
        system: System instruction (persona, rules, context).
    """
    return await generate("therapeutic_chat", prompt, system=system)


async def assess_safety(prompt: str, system: str) -> dict[str, Any]:
    """Safety / alert assessment — always uses Pro model."""
    return await generate_json("safety_analysis", prompt, system=system)