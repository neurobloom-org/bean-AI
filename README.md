# BEAN AI — Privacy-First Mental Health Companion Robot

BEAN (Behavioral Emotional Assistant Node) is a privacy-first AI companion robot designed for teenagers with mental health challenges. The robot (ESP32-S3 hardware) listens to the user, understands their emotional state, holds therapeutic conversations, plays mood-based music, and alerts guardians in crisis situations — all in real time.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [Agents](#5-agents)
6. [API Endpoints](#6-api-endpoints)
7. [WebSocket Protocol (Robot ↔ Server)](#7-websocket-protocol-robot--server)
8. [Services](#8-services)
9. [Background Workers](#9-background-workers)
10. [Database (Supabase)](#10-database-supabase)
11. [Environment Variables](#11-environment-variables)
12. [Deployment (Render)](#12-deployment-render)
13. [ESP32 Hardware Integration](#13-esp32-hardware-integration)
14. [Privacy & Safety Model](#14-privacy--safety-model)
15. [Cost Reference](#15-cost-reference)
16. [Local Development](#16-local-development)

---

## 1. System Overview

```
[ESP32 Robot]
     |
     | WebSocket (wss://)
     | Binary: PCM16 audio (mic input)
     | JSON:   control messages
     |
[BEAN API — Render Standard]
     |
     |-- Deepgram ---------> Speech-to-Text (live transcription)
     |-- Google Gemini -----> AI brain (orchestrator + sub-agents)
     |-- ElevenLabs -------> Text-to-Speech (robot voice, streamed back)
     |-- OpenAI ------------> Embeddings (memory search)
     |-- Supabase ---------> Database, Auth, Storage, Vectors
     |-- Redis ------------> Cache, reminders, rate limiting
     |
[BEAN Workers — Render Starter]
     |-- Reminder check (every 60s) --> Twilio SMS to guardian
     |-- Session cleanup (every 6h)
     |-- Emotion purge (every 6h)
```

The robot sends raw PCM audio over a persistent WebSocket. The server handles everything: transcription, understanding, generating a response, and streaming audio back. The robot only needs to capture audio and play it back.

---

## 2. Architecture

### Conversation Turn Flow

```
1. Robot mic captures voice
2. PCM16 audio chunks sent to server via WebSocket (binary frames)
3. Server forwards chunks to Deepgram in real time
4. Deepgram returns partial + final transcripts
5. On utterance end (1 second silence), orchestrator runs
6. Orchestrator pipeline:
   a. Safety pre-screen (keyword check, no LLM cost)
   b. Route decision (Gemini Flash) → casual / therapy / task / music / alert
   c. Memory context fetch (Supabase pgvector search)
   d. Sub-agent responds (Flash or Pro depending on route)
   e. Post-response: safety score, memory write, transcript store (async)
7. Response text sent to robot as JSON
8. ElevenLabs TTS audio streamed to robot as binary frames
9. Robot plays audio through speaker
```

### Two-Service Split

| Service | Purpose | RAM |
|---|---|---|
| `bean-api` | FastAPI + WebSocket, real-time pipeline | 2 GB (PyTorch) |
| `bean-workers` | Background jobs: reminders, cleanup, purge | 512 MB |

---

## 3. Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| WebSocket | Native FastAPI WebSocket |
| AI Orchestration | Google ADK (Agent Development Kit) |
| LLM | Google Gemini Flash 2.0 (fast) + Gemini 2.5 Pro (therapy/safety) |
| Speech-to-Text | Deepgram nova-2 |
| Text-to-Speech | ElevenLabs Turbo v2 |
| Embeddings | OpenAI text-embedding-3-small |
| Emotion Model | wav2vec2 (HuggingFace, runs locally) |
| Database | Supabase (PostgreSQL + pgvector + Auth + Storage) |
| Cache | Redis (Render Key Value) |
| SMS Alerts | Twilio |
| Deployment | Render (Blueprint) |
| Hardware | ESP32-S3 (WebSocket client) |

---

## 4. Project Structure

```
bean-AI/
├── agents/
│   ├── orchestrator/        # Central coordinator — routes every turn
│   ├── routing/             # Intent classifier (casual/therapy/task/music/alert)
│   ├── casual_chat/         # Fast conversational responses
│   ├── therapeutic_convo/   # CBT/DBT therapeutic responses
│   ├── active_listen/       # Ultra-low-latency filler phrases (<50ms)
│   ├── task/                # Reminder and task creation
│   ├── music/               # Music intent parser (play/stop/next/volume)
│   ├── alert/               # 5-factor safety scoring + guardian SMS
│   ├── emotion/             # wav2vec2 emotion detection on audio
│   ├── memory_writer/       # Extract facts → store embeddings (no raw text)
│   ├── memory/              # Episodic memory retrieval via pgvector
│   ├── stt/                 # Deepgram connection manager
│   └── tts/                 # ElevenLabs TTS with caching
│
├── api/
│   ├── main.py              # App entry point, lifespan, health check
│   ├── websocket_handler.py # Main robot WebSocket endpoint
│   ├── middleware/
│   │   ├── auth_middleware.py   # JWT validation
│   │   └── rate_limiter.py      # Redis-backed rate limiting
│   └── routes/
│       ├── auth.py          # Login, signup, refresh, Google OAuth
│       ├── sessions.py      # Create/retrieve conversation sessions
│       ├── tasks.py         # Reminders and tasks CRUD
│       ├── alerts.py        # Safety alerts list + acknowledge
│       ├── emotions.py      # Emotion history + daily summary
│       ├── guardian.py      # Guardian dashboard (patient oversight)
│       └── internal.py      # Admin endpoints (X-Internal-Key)
│
├── background/
│   ├── worker_main.py       # Entry point for bean-workers service
│   ├── reminder_check.py    # Poll + send due reminders via SMS
│   ├── session_cleanup.py   # Delete expired sessions + transcripts
│   └── emotion_purge.py     # Delete old emotion events
│
├── services/
│   ├── supabase_client.py   # Supabase async client singleton
│   ├── llm_service.py       # Gemini Flash/Pro tiered wrapper
│   ├── deepgram_service.py  # Deepgram WebSocket connection
│   ├── elevenlabs_service.py # TTS streaming + caching
│   ├── embedding_service.py # OpenAI embeddings
│   ├── music_service.py     # Pick songs + stream from Supabase Storage
│   ├── rag_service.py       # CBT/DBT technique retrieval (pgvector)
│   ├── safety_service.py    # Crisis keyword scanner + alert scoring
│   ├── privacy_service.py   # Transcript fetch with TTL enforcement
│   ├── redis_service.py     # Redis async client + helpers
│   ├── calendar_service.py  # Google Calendar OAuth
│   ├── twilio_service.py    # SMS sending + guardian alerts
│   └── cleanup_service.py   # Data retention helpers
│
├── shared/
│   ├── config.py            # All environment variables (Pydantic Settings)
│   ├── schemas.py           # Pydantic models (TokenPayload, UserProfile, etc.)
│   ├── enums.py             # RouteType, AlertLevel, EmotionLabel, etc.
│   ├── exceptions.py        # Custom exception hierarchy
│   └── conversation_cache.py # Redis-backed working memory
│
├── supabase/
│   └── migrations/          # SQL migrations for all tables
│
├── tests/                   # pytest test suite
├── test_esp32_sim.py        # ESP32 hardware simulator (test without robot)
├── pyproject.toml           # Dependencies
├── render.yaml              # Render Blueprint (two-service deployment)
├── Dockerfile               # Production Docker image
└── docker-compose.yml       # Local development environment
```

---

## 5. Agents

All agents are built with Google ADK and run inside the orchestrator pipeline.

### Orchestrator
**File:** `agents/orchestrator/agent.py`

The single entry point for every conversation turn. Never called directly — always invoked by the WebSocket handler.

Pipeline per turn:
1. **Safety pre-screen** — keyword check (no LLM, zero cost)
2. **Route decision** — routing_agent classifies intent
3. **Memory fetch** — recent transcripts + episodic memory (pgvector)
4. **Sub-agent** — responds based on route
5. **Post-response** — async: safety score, memory write, transcript store

### Routing Agent
**File:** `agents/routing/agent.py`

Uses Gemini Flash to classify user intent into one of five routes:

| Route | Trigger |
|---|---|
| `casual` | General conversation, small talk |
| `therapy` | Emotional distress, venting, anxiety |
| `task` | "Remind me to...", "Set a task for..." |
| `music` | "Play music", "Stop the music", "Something calm" |
| `alert` | Crisis signals (overrides routing if safety threshold met) |

Alert route requires confidence ≥ 0.85. Low-confidence alert downgrades to `therapy`.

### Casual Chat Agent
**File:** `agents/casual_chat/agent.py`

- Model: Gemini Flash
- Max: 2 sentences, 260 characters
- Persona-driven (warm, teen-friendly)
- Output sanitized: no emoji, no asterisks, no roleplay actions

### Therapy Agent
**File:** `agents/therapeutic_convo/agent.py`

- Model: Gemini Pro (best quality)
- Injects 3 relevant CBT/DBT techniques via RAG (pgvector search)
- Uses user profile + emotion trend in system prompt
- 250-token max (keeps responses concise and grounded)

### Active Listen Agent
**File:** `agents/active_listen/agent.py`

- < 50ms response time
- No LLM calls — rule-based emotion → phrase mapping
- Fills the latency gap while Gemini generates a full response
- Example: detects "sad" → plays "I hear you..." filler phrase immediately

### Task Agent
**File:** `agents/task/agent.py`

- Two-phase: extract draft → confirm → save
- Parses: title, description, due_at, reminder_at
- Stores in Supabase `tasks` table
- Reminder at due time: pushed to robot via WebSocket if connected, SMS fallback

### Music Agent
**File:** `agents/music/agent.py`

Parses music intent only. Actual streaming is handled by `MusicPlayer` in the WebSocket handler.

| Action | Trigger |
|---|---|
| `play_music` | "Play something calm", "Put on music" |
| `stop_music` | "Stop the music" |
| `next_track` | "Skip this song", "Play something else" |
| `pause_music` | "Pause" |
| `resume_music` | "Resume" |
| `set_volume` | "Turn it up", "Volume 50" |

Moods: `calm`, `happy`, `sad`, `lofi`, `nature`, `classical`

### Alert Agent
**File:** `agents/alert/agent.py`

5-factor safety scoring runs on every turn:

| Factor | Signal |
|---|---|
| F1 | Crisis keyword detected (kill myself, suicide, self harm...) |
| F2 | Negative emotion detected (angry, sad, fearful) |
| F3 | Escalation pattern (getting worse over turns) |
| F4 | Vulnerability flag (user is a minor) |
| F5 | Explicit statement ("I will...", "I already took...") |

Thresholds: Adults need 3 factors. Minors need 2 (F4 is pre-counted).
On threshold: sends SMS to guardian via Twilio, creates alert record, 5-minute cooldown.

### Emotion Agent
**File:** `agents/emotion/agent.py`

- Runs wav2vec2 model locally on PCM audio
- Temporal smoothing via exponential moving average (EMA)
- Dynamic noise-floor gating per session
- Neutral-biased prior to avoid cold-start false positives
- Labels: `angry`, `sad`, `fearful`, `disgusted`, `surprised`, `happy`, `calm`, `neutral`

### Memory Writer Agent
**File:** `agents/memory_writer/agent.py`

- Runs after each turn (fire-and-forget, non-blocking)
- Extracts structured facts from conversation via LLM
- Strips PII before embedding
- Stores only vector embeddings — never raw conversation text
- Upserts `user_profiles` and `episodic_memory` tables

---

## 6. API Endpoints

All endpoints require `Authorization: Bearer <JWT>` unless noted.

### Authentication
```
POST   /api/v1/auth/login                  — email + password → JWT
POST   /api/v1/auth/signup                 — create account
POST   /api/v1/auth/refresh                — refresh token → new JWT
GET    /api/v1/auth/google                 — begin Google Calendar OAuth
GET    /api/v1/auth/google/callback        — OAuth redirect (public)
```

### Sessions
```
POST   /api/v1/sessions                    — create conversation session
GET    /api/v1/sessions/{session_id}       — get session metadata
```

### Tasks & Reminders
```
POST   /api/v1/tasks                       — create reminder task
GET    /api/v1/tasks?status=pending        — list tasks (filter by status)
```

### Alerts
```
GET    /api/v1/alerts                      — list safety alerts
PATCH  /api/v1/alerts/{alert_id}/acknowledge — mark alert as read
```

### Emotions
```
GET    /api/v1/emotions?days=7             — emotion events history
GET    /api/v1/emotions/summary            — daily aggregated summary
```

### Guardian Dashboard
```
GET    /api/v1/guardian/patients                              — list linked patients
GET    /api/v1/guardian/patients/{patient_id}/emotions        — patient emotion trends
```

Guardians see aggregated labels only — never raw transcripts.

### Health Check (public)
```
GET    /api/v1/health                      — Supabase + Redis status
```

### Internal (X-Internal-Key required)
```
POST   /internal/purge                     — trigger data retention cleanup
GET    /internal/health                    — detailed health status
```

---

## 7. WebSocket Protocol (Robot ↔ Server)

### Connection

```
wss://bean-api-jolw.onrender.com/ws?device_id=<YOUR_DEVICE_ID>
```

The `device_id` must be pre-registered in the Supabase `devices` table with a linked `user_id`.

Alternatively (for testing), pass a JWT:
```
Header: Sec-WebSocket-Protocol: bearer.<your_jwt_token>
```

### On Connect

Server immediately sends:
```json
{"type": "connected", "session_id": "uuid-here"}
```

### Robot → Server Messages

**Binary frame:** Raw PCM16 audio
- Format: 16-bit signed integer, mono, 16kHz sample rate
- Send continuously while user is speaking
- Recommended chunk size: ~3.2 KB (200ms of audio)

**JSON control messages:**
```json
{"type": "ping"}
{"type": "emotion_result", "emotion": "sad", "confidence": 0.85}
{"type": "music_status", "status": "playing", "genre": "calm"}
{"type": "robot_status", "battery_level": 85, "wifi_rssi": -60}
{"type": "end_session"}
```

### Server → Robot Messages

**Transcription feedback:**
```json
{"type": "transcript_partial", "text": "I feel so..."}
{"type": "transcript_final", "text": "I feel so alone.", "confidence": 0.97}
```

**AI response:**
```json
{"type": "response_text", "text": "I'm here with you.", "route": "therapy", "turn_id": "uuid"}
```

**TTS audio (binary + JSON sentinels):**
```
{"type": "tts_start", "turn_id": "uuid"}
<binary audio chunk>
<binary audio chunk>
...
{"type": "tts_end", "turn_id": "uuid"}
```

**Music streaming:**
```json
{"type": "music_start", "song_id": "uuid", "title": "Calm Rain", "mood": "calm"}
```
```
<binary 8KB audio chunk>
<binary 8KB audio chunk>
...
```
```json
{"type": "music_end", "song_id": "uuid"}
{"type": "music_stopped"}
{"type": "music_volume", "volume": 70}
{"type": "music_unavailable", "mood": "calm", "message": "No calm songs available yet."}
```

**Keepalive:**
```json
{"type": "ping", "timestamp": "2026-03-29T05:50:47Z"}
```
Robot should respond:
```json
{"type": "pong", "timestamp": "..."}
```

**Errors:**
```json
{"type": "error", "code": "stt_unavailable", "message": "..."}
{"type": "error", "code": "rate_limited", "message": "Too many messages — please slow down."}
{"type": "error", "code": "processing_failed", "message": "Something went wrong — please try again."}
{"type": "error", "code": "tts_failed", "message": "Audio unavailable."}
```

### Full Conversation Turn Sequence

```
Robot                              BEAN API
  |                                    |
  |── connect wss://.../ws ───────────▶|
  |◀─ {"type":"connected"} ────────────|
  |                                    |── Deepgram STT connect
  |                                    |
  |── [PCM16 binary chunks] ──────────▶|──▶ Deepgram
  |── [PCM16 binary chunks] ──────────▶|
  |◀─ {"type":"transcript_partial"} ───|◀── partial text
  |── [PCM16 binary chunks] ──────────▶|
  |◀─ {"type":"transcript_final"} ─────|◀── final text
  |                                    |
  |                                    |── [1s silence detected]
  |                                    |── Orchestrator runs
  |                                    |── Gemini Flash (route)
  |                                    |── Gemini Pro (therapy)
  |                                    |── [async: safety, memory]
  |                                    |
  |◀─ {"type":"response_text"} ────────|
  |◀─ {"type":"tts_start"} ────────────|──▶ ElevenLabs TTS
  |◀─ [binary audio] ──────────────────|◀── audio stream
  |◀─ [binary audio] ──────────────────|
  |◀─ {"type":"tts_end"} ──────────────|
  |                                    |
  |── plays audio on speaker           |
```

### ESP32 Firmware Checklist

```
1. Connect WebSocket with device_id query param
2. Send PCM16 chunks continuously from mic while user speaks
3. On {"type": "tts_start"}: mute mic (don't send audio while robot speaks)
4. On binary frames received: buffer and play through speaker
5. On {"type": "tts_end"}: unmute mic, resume recording
6. On {"type": "music_start"}: prepare to receive and play binary chunks
7. On {"type": "music_stopped"}: stop speaker
8. Send {"type": "ping"} every 30s to keep connection alive
9. On {"type": "pong"}: connection confirmed alive
10. Send {"type": "robot_status", "battery_level": N, "wifi_rssi": N} periodically
```

### Device Registration

Before the robot can connect, register it in Supabase:

```sql
INSERT INTO devices (device_id, user_id, device_name)
VALUES ('ESP32_BEAN_001', '<user-uuid>', 'BEAN Robot');
```

---

## 8. Services

### LLM Service (`services/llm_service.py`)
Two-tier Gemini routing:

| Tier | Model | Used for | Cost |
|---|---|---|---|
| Flash | gemini-2.0-flash | routing, casual, task, music, memory | ~$0.00001/call |
| Pro | gemini-2.5-pro | therapy, safety, crisis | ~$0.004/call |

85% cheaper than all-Pro tier.

### Deepgram Service (`services/deepgram_service.py`)
- Persistent WebSocket per session (not per audio chunk)
- PCM16, 16kHz, mono, nova-2 model
- Endpointing: 500ms silence → partial final, 1000ms → utterance end
- Auto-reconnect: up to 3 attempts (2s, 4s, 6s backoff)
- Privacy: audio exists only in RAM transit, never written to disk

### ElevenLabs Service (`services/elevenlabs_service.py`)
- Turbo v2 model (low latency)
- Streams audio chunks directly to robot WebSocket
- TTS cache: SHA-256(voice_id + text) → Supabase (7-day TTL, avoids re-generating same phrases)
- Timeouts: 10s first chunk, 8s per-chunk, 30s total

### Music Service (`services/music_service.py`)
- Songs stored in Supabase Storage bucket `bean-music`
- Served via 5-minute signed URLs
- Streamed in 8KB chunks to robot
- Picks user-uploaded songs first, falls back to default library
- Supported: MP3, OGG, WAV, M4A, AAC (max 20MB per file)

### RAG Service (`services/rag_service.py`)
- Retrieves CBT/DBT therapeutic techniques via pgvector similarity search
- Enriches query: emotion label + user text → embedding → top-3 results
- Injected into therapy agent system prompt per turn
- Minimum similarity: 0.5

### Safety Service (`services/safety_service.py`)
- 13 crisis keywords (fast pre-screen, no LLM)
- 9 explicit statement patterns (F5 factor)
- 5-factor scoring system
- Post-alert support messages (3 rotating templates)

---

## 9. Background Workers

Workers run in `bean-workers` (separate Render service). They never handle HTTP or WebSocket.

### Reminder Check (`background/reminder_check.py`)
- Runs every 60 seconds
- Queries Supabase for tasks where `reminder_at <= NOW()` and `status = pending`
- Staleness guard: skips tasks > 2 hours overdue (restart safety)
- Delivery: pushes to active WebSocket session first, falls back to Twilio SMS
- On no response: 30-minute snooze

### Session Cleanup (`background/session_cleanup.py`)
- Runs every 6 hours
- Deletes sessions older than 30 days
- Purges session_transcripts older than 24 hours
- Cleans expired rate limit counters

### Emotion Purge (`background/emotion_purge.py`)
- Runs every 6 hours
- Deletes emotion_events older than 90 days

---

## 10. Database (Supabase)

Key tables:

| Table | Purpose |
|---|---|
| `sessions` | Conversation session records |
| `session_transcripts` | Turn-by-turn text (24h TTL) |
| `emotion_events` | Raw emotion detections (90d TTL) |
| `tasks` | User reminders and tasks |
| `alerts` | Safety alert records |
| `devices` | Registered ESP32 devices |
| `user_profiles` | Display name, preferences, diagnosis tags |
| `episodic_memory` | Vector embeddings of conversation facts |
| `guardian_links` | Guardian ↔ patient relationships |
| `tts_cache` | Cached TTS audio (7d TTL) |
| `therapeutic_techniques` | CBT/DBT technique library (pgvector) |
| `songs` | Music library metadata |

Row Level Security (RLS) is enabled on all user-facing tables. The service role key (used by the backend) bypasses RLS for admin operations. User JWTs go through RLS enforcement.

---

## 11. Environment Variables

All variables set in Render dashboard (Environment tab) for each service.

### Required — Both Services
```
ENVIRONMENT=production
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_SERVICE_ROLE_KEY=eyJ...
SUPABASE_JWT_SECRET=your-jwt-secret
REDIS_URL=redis://red-xxxx:6379
```

### Required — bean-api Only
```
SUPABASE_ANON_KEY=eyJ...
GOOGLE_API_KEY=AIza...
OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=pNInz6obpgDQGcFmaJgB
RATE_LIMIT_HASH_SALT=<32-char hex>
OAUTH_STATE_SECRET=<32-char hex>
INTERNAL_API_KEY=<32-char hex>
CORS_ALLOWED_ORIGINS=https://your-frontend.com
FRONTEND_BASE_URL=https://your-frontend.com
GOOGLE_OAUTH_CLIENT_ID=...
GOOGLE_OAUTH_CLIENT_SECRET=...
GOOGLE_OAUTH_REDIRECT_URI=https://bean-api-jolw.onrender.com/api/v1/auth/google/callback
RUN_BACKGROUND_WORKERS=false
```

### Required — bean-workers Only
```
RUN_BACKGROUND_WORKERS=true
TWILIO_ACCOUNT_SID=AC...
TWILIO_AUTH_TOKEN=...
TWILIO_FROM_NUMBER=+1...
```

### Generate Secret Values
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## 12. Deployment (Render)

### Services

| Service | Type | Plan | Cost |
|---|---|---|---|
| bean-api | Web Service | Standard (2GB RAM) | ~$25/mo |
| bean-workers | Worker | Starter (512MB) | ~$7/mo |
| bean-redis | Key Value | Free | $0 |

bean-api requires Standard plan because wav2vec2 + PyTorch load ~1.5GB RAM.

### Deploy from Blueprint

The `render.yaml` file in the repo root is a Blueprint that defines both services. Connect the repo in the Render dashboard and it will create both automatically.

### Health Check

```bash
curl https://bean-api-jolw.onrender.com/api/v1/health
```

Expected response when fully operational:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production",
  "supabase": true,
  "redis": true,
  "deepgram_configured": true,
  "elevenlabs_configured": true,
  "gemini_configured": true
}
```

---

## 13. ESP32 Hardware Integration

### Audio Requirements
- **Format:** PCM16 (16-bit signed integer)
- **Sample rate:** 16,000 Hz (16kHz) — must match exactly
- **Channels:** Mono (1 channel)
- **Chunk size:** ~3.2 KB per 200ms recommended

### WebSocket URL
```
wss://bean-api-jolw.onrender.com/ws?device_id=YOUR_DEVICE_ID
```

### Minimal Arduino/ESP-IDF Pseudocode
```cpp
// 1. Connect WiFi
// 2. Open WebSocket
ws.connect("wss://bean-api-jolw.onrender.com/ws?device_id=ESP32_BEAN_001");

// 3. On connect, wait for {"type":"connected"}

// 4. Capture mic and stream
while (session_active) {
    int16_t audio[3200]; // 200ms at 16kHz
    mic.read(audio, sizeof(audio));
    ws.send_binary(audio, sizeof(audio));
}

// 5. On binary received → play on speaker
ws.on_binary([](uint8_t* data, size_t len) {
    speaker.play(data, len);
});

// 6. On JSON received → parse type field
ws.on_text([](String msg) {
    if (type == "tts_start") mic.mute();
    if (type == "tts_end")   mic.unmute();
    if (type == "music_start") display.show(title);
    if (type == "music_stopped") speaker.stop();
});
```

### Testing Without Hardware

Use the built-in simulator:
```bash
python test_esp32_sim.py --email user@example.com --password yourpassword
# or with a WAV file
python test_esp32_sim.py --email user@example.com --password yourpassword --wav test.wav
```

---

## 14. Privacy & Safety Model

### Privacy Principles
- Audio is never written to disk — exists only in RAM during transit
- Transcripts are deleted after 24 hours
- Embeddings stored without source text (PII-stripped before embedding)
- Guardians see aggregated emotion labels only — never raw conversation content
- All secrets via environment variables — never hardcoded

### Safety System
The 5-factor alert model scores every turn:

```
F1: Crisis keyword detected       (kill myself, suicide, self harm, ...)
F2: Negative emotion detected     (angry, sad, fearful, disgusted)
F3: Escalation pattern            (emotion getting worse over last 5 turns)
F4: Vulnerability flag            (user is flagged as minor)
F5: Explicit statement            (I will..., I already took..., I have a plan...)

Adult threshold:  3 factors → HIGH alert
Minor threshold:  2 factors → HIGH alert (F4 pre-counted = 1)
```

On alert:
1. Alert record saved to Supabase
2. SMS sent to registered guardian via Twilio
3. Support message sent to robot
4. 5-minute cooldown before next alert

---

## 15. Cost Reference

### Per Conversation (~5 minutes)

| Service | Usage | Approx Cost |
|---|---|---|
| Deepgram STT | 5 min audio | ~$0.020 |
| Gemini Flash | routing + casual | ~$0.001 |
| Gemini Pro | 1-2 therapy turns | ~$0.008 |
| ElevenLabs TTS | ~500 chars | ~$0.005 |
| OpenAI Embeddings | ~1k tokens | ~$0.0001 |
| **Total per conversation** | | **~$0.034** |

### Monthly Infrastructure

| Service | Cost |
|---|---|
| Render bean-api (Standard) | $25/mo |
| Render bean-workers (Starter) | $7/mo |
| Render Redis (Free) | $0 |
| **Total** | **~$32/mo** |

---

## 16. Local Development

### Prerequisites
- Python 3.12
- Redis running locally
- Supabase project (or local Supabase via Docker)
- API keys for Deepgram, Gemini, ElevenLabs, OpenAI

### Setup
```bash
# Clone
git clone https://github.com/neurobloom-org/bean-AI.git
cd bean-AI

# Install dependencies
pip install -e ".[dev]"

# Copy env template
cp .env.example .env
# Fill in your API keys in .env

# Run API server
uvicorn api.main:app --reload --port 8080

# Run workers (separate terminal)
python -m background.worker_main
```

### Using Docker Compose
```bash
docker compose up
```

### Run Tests
```bash
pytest tests/
```

### Simulate the Robot
```bash
python test_esp32_sim.py --email your@email.com --password yourpass
```

---

## Live Deployment

- **API:** https://bean-api-jolw.onrender.com
- **WebSocket:** wss://bean-api-jolw.onrender.com/ws
- **Health:** https://bean-api-jolw.onrender.com/api/v1/health
- **Branch:** `feature/fixing`
