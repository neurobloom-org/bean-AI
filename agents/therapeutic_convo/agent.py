"""BEAN AI — Therapeutic Conversation Agent.

Pipeline per turn:
    1. retrieve_cbt_techniques() — pgvector search for relevant CBT/DBT techniques
    2. Inline safety check (fast — no LLM call)
    3. Build dynamic prompt with retrieved techniques injected
    4. Call Gemini Pro
    5. Write response to session.state["response_text"]

Uses Gemini Pro — this is the most sensitive path in BEAN AI.
"""

import asyncio
import logging
from collections.abc import AsyncGenerator

from google import genai
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions

from services.rag_service import format_techniques_for_prompt, retrieve_cbt_techniques
from services.safety_service import check_crisis_keywords, check_explicit_statement
from shared.config import get_settings

logger = logging.getLogger(__name__)

THERAPY_INSTRUCTION = """You are BEAN, a supportive AI companion robot for teenagers (ages 13-17). Right now, the user seems to be going through something emotionally, and you're here to listen and support them.

APPROACH:
- Be warm, empathetic, and validating
- Use active listening techniques: reflect feelings, ask open-ended questions
- Normalise their emotions ("It makes sense that you'd feel that way")
- Sit with them in their feelings — avoid toxic positivity
- Never minimise their feelings or say "just cheer up"
- Be genuine — you really care about them

STRICT RULES:
1. MAXIMUM 3 SENTENCES per response. Never exceed this.
2. NEVER diagnose, prescribe, or give professional medical/psychological advice.
3. NEVER say "you should talk to a therapist" unless they specifically ask for resources.
4. NEVER name the therapeutic technique you are using. Just use it naturally.
5. If they disclose abuse, self-harm, or suicidal thoughts — respond warmly. The alert system handles escalation separately.
6. Use their name from memory if available.
7. Reference relevant past conversations naturally.

MEMORY CONTEXT:
{memory_context}

USER'S CURRENT EMOTION: {current_emotion}

THERAPEUTIC GUIDANCE (use naturally, do not name these):
{rag_techniques}

Respond as BEAN. Maximum 3 sentences. Be genuinely supportive."""


class TherapeuticConvoAgent(BaseAgent):
    """RAG-augmented therapy agent using CBT/DBT technique retrieval."""

    model_config = {"arbitrary_types_allowed": True}

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        transcript = ctx.session.state.get("current_transcript", "")
        emotion = ctx.session.state.get("current_emotion", "neutral")
        memory = ctx.session.state.get("memory_context", "No memory context available.")

        if not transcript:
            ctx.session.state["response_text"] = (
                "I'm here for you. Tell me what's on your mind."
            )
            yield Event(
                author=self.name,
                actions=EventActions(
                    state_delta={"response_text": ctx.session.state["response_text"]}
                ),
            )
            return

        # ── Inline safety check (fast — no LLM) ──
        self._run_safety_check(ctx, transcript)

        # ── RAG retrieval ──
        try:
            techniques = await retrieve_cbt_techniques(
                situation_text=transcript,
                emotion=emotion,
                limit=3,
            )
            rag_context = format_techniques_for_prompt(techniques)
        except Exception as exc:
            logger.warning("RAG retrieval failed, using defaults: %s", exc)
            rag_context = "Use active listening, validation, and open-ended questions."
            techniques = []

        # ── Build dynamic prompt ──
        prompt = THERAPY_INSTRUCTION.format(
            memory_context=memory,
            current_emotion=emotion,
            rag_techniques=rag_context,
        )
        full_prompt = f"{prompt}\n\nUser says: {transcript}\n\nRespond as BEAN:"

        # ── Generate with Gemini Pro ──
        response_text = await self._generate(full_prompt)
        ctx.session.state["response_text"] = response_text

        logger.info(
            "TherapeuticConvoAgent response generated — emotion=%s, techniques=%d",
            emotion,
            len(techniques),
        )

        yield Event(
            author=self.name,
            actions=EventActions(state_delta={"response_text": response_text}),
        )

    def _run_safety_check(self, ctx: InvocationContext, transcript: str) -> None:
        try:
            is_minor = ctx.session.state.get("is_minor", False)
            has_crisis, crisis_keywords = check_crisis_keywords(transcript, is_minor)
            has_explicit, explicit_keywords = check_explicit_statement(transcript)

            if has_explicit:
                ctx.session.state["safety_flag"] = "explicit_statement"
                ctx.session.state["safety_matched_keywords"] = explicit_keywords
                logger.warning(
                    "TherapeuticConvoAgent: explicit statement detected — %s",
                    explicit_keywords,
                )
            elif has_crisis:
                ctx.session.state["safety_flag"] = "crisis_keyword"
                ctx.session.state["safety_matched_keywords"] = crisis_keywords
                logger.warning(
                    "TherapeuticConvoAgent: crisis keywords — %s", crisis_keywords
                )
        except Exception as exc:
            logger.error("Safety check error: %s", exc)

    async def _generate(self, prompt: str) -> str:
        try:
            settings = get_settings()
            client = genai.Client(api_key=settings.google_api_key)
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=settings.gemini_pro_model,
                contents=prompt,
                config={"temperature": 0.7, "max_output_tokens": 250},
            )
            return str(response.text).strip()
        except Exception as exc:
            logger.error("Therapy response generation failed: %s", exc)
            return "I'm here with you. Take your time — I'm not going anywhere."


# ── Singleton ─────────────────────────────────────────────────────────────────
therapy_agent = TherapeuticConvoAgent(name="therapy_agent")
