# BEAN AI — System Architecture

## Overview

BEAN is a real-time voice pipeline backend for a companion robot serving teenagers (ages 13–17). A robot connects over WebSocket, streams audio, and receives spoken responses — all within a latency budget tight enough for natural conversation.

The backend is built on **Google ADK** (Agent Development Kit) with **FastAPI** as the HTTP/WebSocket server, deployed on **Google Cloud Run**.

---

## High-Level Pipeline

```
Robot (hardware)
    │
    │  WebSocket (audio frames in, text+audio out)
    ▼
api/websocket_handler.py
    │
    │  Authenticates token, creates session, starts event loop
    ▼
BEANOrchestrator  (agents/orchestrator/agent.py)
    │
    ├─── PHASE 1: Parallel  ──────────────────────────────────────
    │         ├── AlertAgent          (safety monitoring)
    │         └── MemoryAgent         (Redis + pgvector context)
    │
    ├─── PHASE 2: Routing  ───────────────────────────────────────
    │         └── RoutingAgent        (Gemini Flash, temp=0)
    │
    ├─── PHASE 3: Response  ──────────────────────────────────────
    │         └── [one of]
    │               ├── CasualChatAgent       (Gemini Flash)
    │               ├── TherapeuticConvoAgent (Gemini Pro)
    │               ├── TaskAgent             (Gemini Flash + Calendar)
    │               └── MusicAgent            (Gemini Flash + WS commands)
    │
    └─── PHASE 4: TTS + Delivery  ───────────────────────────────
              └── ElevenLabs Turbo v2  →  audio chunks → robot
```

---

## Per-Turn Sequence

Each complete user utterance (detected by Deepgram's endpointing) triggers one full pipeline turn:

```
1. Deepgram fires utterance_end callback
2. Transcript written to session.state["current_transcript"]
3. Emotion written to session.state["current_emotion"] (from parallel wav2vec2)
4. BEANOrchestrator._run_async_impl() called

   ┌─ asyncio.gather() ──────────────────────┐
   │  AlertAgent._run_async_impl()           │  ← evaluates 5-factor safety score
   │  MemoryAgent._run_async_impl()          │  ← Redis working mem + pgvector
   └─────────────────────────────────────────┘

5. RoutingAgent classifies intent → route string
6. Response agent selected by route, generates text
7. Diversity check (cache_service) — retry once if fails
8. WorkingMemoryEntry written to Redis (user + assistant turns)
9. TTS: ElevenLabs streams 4096-byte audio chunks to robot
10. Event yielded with state_delta (response_text, route, latency_ms)
```

---

## ADK Agent Types Used

| Agent | ADK Type | Notes |
|-------|----------|-------|
| BEANOrchestrator | `BaseAgent` (custom) | Custom `_run_async_impl` for WebSocket event loop |
| AlertAgent | `BaseAgent` (custom) | Parallel safety evaluation, no LLM |
| MemoryAgent | `BaseAgent` (custom) | Parallel memory retrieval, no LLM |
| RoutingAgent | `LlmAgent` | Gemini Flash, `temperature=0`, max 10 tokens |
| CasualChatAgent | `LlmAgent` | Gemini Flash, `temperature=0.8`, 2-sentence max |
| TherapeuticConvoAgent | `LlmAgent` | Gemini Pro, 3-sentence max, safety callback |
| TaskAgent | `LlmAgent` | Gemini Flash + Google Calendar FunctionTool |
| MusicAgent | `LlmAgent` | Gemini Flash + WebSocket command FunctionTool |
| STT | `FunctionTool` | Wraps Deepgram Nova-2 WebSocket, not an LlmAgent |
| Emotion | `FunctionTool` | wav2vec2 inference, not an LlmAgent |
| TTS | `FunctionTool` | Wraps ElevenLabs Turbo v2 streaming, not an LlmAgent |
| ActiveListen | `FunctionTool` | Returns filler phrases while pipeline processes |

---

## Session State Contract

`ctx.session.state` is the shared message bus between agents within a turn. Key fields:

| Key | Set by | Read by |
|-----|--------|---------|
| `session_id` | websocket_handler | all agents |
| `user_id` | websocket_handler | all agents |
| `current_transcript` | websocket_handler (Deepgram callback) | orchestrator, alert, routing |
| `current_emotion` | emotion FunctionTool | orchestrator, alert, routing, response agents |
| `emotion_confidence` | emotion FunctionTool | alert agent |
| `is_minor` | websocket_handler (from DB) | alert agent |
| `memory_context` | MemoryAgent | routing agent, all response agents |
| `route` | orchestrator (from RoutingAgent) | orchestrator |
| `alert_state` | AlertAgent | orchestrator |
| `alert_dispatched` | AlertAgent | orchestrator |
| `post_alert_message` | AlertAgent | orchestrator (appended to response) |
| `response_text` | response agent | orchestrator, TTS |
| `latency_ms` | orchestrator | logging, metrics |

---

## Infrastructure

```
Google Cloud Run
    └── FastAPI (uvicorn, port 8080)
            ├── /ws              WebSocket endpoint
            ├── /api/sessions    REST: session management
            ├── /api/tasks       REST: task/reminder CRUD
            ├── /api/alerts      REST: alert log access
            └── /api/emotions    REST: emotion history

Cloud SQL (PostgreSQL 15 + pgvector)
    └── 8 tables (see deployment.md for schema)

Cloud Memorystore (Redis 7)
    ├── Working memory  (Redis List, last 5 turns, per user)
    ├── Semantic profile (Redis JSON, persistent)
    ├── TTS cache        (7-day TTL)
    └── Response cache   (1-hour TTL, diversity check)

External APIs
    ├── Deepgram Nova-2     (streaming STT, WebSocket)
    ├── ElevenLabs Turbo v2 (streaming TTS)
    ├── OpenAI              (text-embedding-3-small)
    ├── Twilio              (SMS guardian alerts)
    └── Google Calendar v3  (task agent)
```

---

## Why BaseAgent Instead of Standard ADK Workflow Agents

The real-time voice pipeline requires a persistent WebSocket connection and a custom event loop that:

- Handles continuous audio frame ingestion outside of ADK's standard request/response model
- Manages per-session Deepgram WebSocket connections
- Runs parallel async tasks (alert + memory) without ADK's built-in ParallelAgent routing overhead
- Streams TTS audio chunks back mid-turn

ADK's `SequentialAgent` and `ParallelAgent` are designed for discrete, complete invocations. `BEANOrchestrator` uses `BaseAgent` with a custom `_run_async_impl` to keep full control of the async event loop.
