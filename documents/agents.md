# BEAN AI — Agents Reference

All agents live under `agents/`. Each has its own directory with `agent.py`, `__init__.py`, `.env`, `README.md`, and `test_agent.py`.

---

## BEANOrchestrator

**File:** `agents/orchestrator/agent.py`  
**Type:** `BaseAgent` (custom `_run_async_impl`)  
**Model:** None (no LLM — orchestration only)

The main coordinator. Owns the WebSocket event loop and dispatches all other agents. One instance per server process; per-session state lives in `ctx.session.state`.

**Responsibilities:**
- Runs AlertAgent + MemoryAgent in parallel via `asyncio.gather()`
- Invokes RoutingAgent to classify intent
- Selects and runs the appropriate response agent
- Runs diversity check; retries once if response fails
- Writes both user and assistant turns to Redis working memory
- Streams TTS audio chunks back to robot

**Env vars:** `GOOGLE_API_KEY`, `DATABASE_URL`, `REDIS_URL`

---

## AlertAgent

**File:** `agents/alert/agent.py`  
**Type:** `BaseAgent` (custom)  
**Model:** None (rule-based + keyword matching)

Runs in parallel with MemoryAgent on every turn. Evaluates the 5-factor safety score and dispatches an SMS to the guardian when the threshold is met. See [safety-system.md](./safety-system.md) for full details.

**Inputs:** `current_transcript`, `current_emotion`, `emotion_confidence`, `is_minor`  
**Outputs:** `alert_state`, `alert_dispatched`, `post_alert_message`  
**Env vars:** `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_FROM_NUMBER`

---

## MemoryAgent

**File:** `agents/memory/agent.py`  
**Type:** `BaseAgent` (custom)  
**Model:** None (data retrieval only)

Runs three memory lookups in parallel via `asyncio.gather()` and assembles a `MemoryContext` string injected into all response agent prompts.

**Memory sources:**

| Source | Storage | Content | TTL |
|--------|---------|---------|-----|
| Working memory | Redis List | Last 5 turns (user + assistant) | 24 hours |
| Semantic profile | Redis JSON | Persistent user facts, preferences, name | No expiry |
| Episodic memory | pgvector | Relevant past conversations (cosine similarity ≥ 0.3) | 30 days |

**Inputs:** `user_id`, `current_transcript`  
**Outputs:** `memory_context` (formatted string injected into LLM prompts)  
**Env vars:** `REDIS_URL`, `DATABASE_URL`, `OPENAI_API_KEY` (for embeddings)

---

## RoutingAgent

**File:** `agents/routing/agent.py`  
**Type:** `LlmAgent`  
**Model:** Gemini 2.5 Flash, `temperature=0.0`, max 10 tokens

Classifies user intent into exactly one of five routes. Uses temperature=0 for deterministic, consistent routing.

**Routes (priority order):**

| Route | When | Priority |
|-------|------|----------|
| `alert` | Explicit self-harm / crisis statements | Highest |
| `task` | Reminders, schedules, calendar | 2 |
| `music` | Music playback, songs, playlists | 3 |
| `therapy` | Emotional distress, seeking support | 4 |
| `casual` | Everything else | Default |

**Output:** Single word route name → parsed by `parse_route()` into `RouteType` enum. Defaults to `casual` on unrecognised output.

**Env vars:** `GOOGLE_API_KEY`

---

## CasualChatAgent

**File:** `agents/casual_chat/agent.py`  
**Type:** `LlmAgent`  
**Model:** Gemini 2.5 Flash, `temperature=0.8`

BEAN's default conversational voice. Warm, casual, like a cool older friend — not a therapist, not a corporate bot.

**Constraints:**
- Maximum **2 sentences** per response
- Uses `{memory_context}` and `{current_emotion}` template vars
- Never says "As an AI" — always speaks as BEAN

**Env vars:** `GOOGLE_API_KEY`

---

## TherapeuticConvoAgent

**File:** `agents/therapeutic_convo/agent.py`  
**Type:** `LlmAgent`  
**Model:** Gemini 2.5 **Pro**, `temperature=0.7`

Activated when the router classifies emotional distress. Uses Gemini Pro (higher quality) for more nuanced supportive responses. Has a `before_model_callback` that runs `SafetyService` crisis checks inline before generation.

**Constraints:**
- Maximum **3 sentences** per response
- Active listening techniques: reflect feelings, open-ended questions
- Never diagnoses, prescribes, or refers to professional therapy unprompted
- Avoids toxic positivity

**Env vars:** `GOOGLE_API_KEY`

---

## TaskAgent

**File:** `agents/task/agent.py`  
**Type:** `LlmAgent`  
**Model:** Gemini 2.5 Flash

Handles reminders, calendar events, and task management. Has access to Google Calendar API via FunctionTool.

**FunctionTools:** `create_reminder`, `list_reminders`, `delete_reminder`, Google Calendar read/write  
**Env vars:** `GOOGLE_API_KEY`, `GOOGLE_CALENDAR_CREDENTIALS`

---

## MusicAgent

**File:** `agents/music/agent.py`  
**Type:** `LlmAgent`  
**Model:** Gemini 2.5 Flash

Handles music playback control. Generates structured WebSocket command messages that the robot hardware interprets.

**Commands:** play, pause, skip, volume_up, volume_down, play_mood_playlist  
**Env vars:** `GOOGLE_API_KEY`

---

## STT (Deepgram)

**File:** `agents/stt/agent.py`  
**Type:** `FunctionTool` (not an LlmAgent)  
**Service:** Deepgram Nova-2

Wraps a persistent per-session Deepgram WebSocket connection. Audio frames are sent via `deepgram_transcribe()`. Transcripts arrive via callback and are written to `session.state["current_transcript"]`.

**Config:** 16kHz, mono, linear16 PCM, 500ms endpointing, 1000ms utterance-end detection  
**Env vars:** `DEEPGRAM_API_KEY`

---

## Emotion (wav2vec2)

**File:** `agents/emotion/agent.py`  
**Type:** `FunctionTool` (not an LlmAgent)  
**Model:** wav2vec2 (CPU inference, lazy-loaded)

Classifies emotion from 500ms PCM audio windows. Returns one of 8 labels: `angry`, `calm`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`.

**Input:** Base64-encoded PCM int16, 16kHz  
**Output:** `EmotionResult { label, confidence }`  
**Notes:** Model lazy-loads on first call. CPU only — no GPU required.  
**Env vars:** `EMOTION_MODEL_NAME` (HuggingFace model ID)

---

## TTS (ElevenLabs)

**File:** `agents/tts/agent.py`  
**Type:** `FunctionTool` (not an LlmAgent)  
**Service:** ElevenLabs Turbo v2

Synthesizes speech from text and streams 4096-byte audio chunks back to the robot over WebSocket. Results cached in Redis for 7 days (common phrases like filler phrases are pre-cached on startup).

**Env vars:** `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`

---

## ActiveListenAgent

**File:** `agents/active_listen/agent.py`  
**Type:** `FunctionTool` (not an LlmAgent)

Returns context-appropriate filler phrases ("Hmm, let me think about that...") played to the robot while the main pipeline processes. Keeps the interaction feeling natural during processing latency.

**No external API calls.** Phrase selection is rule-based on current emotion and route.
