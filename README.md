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
17. [Performance Design](#17-performance-design)

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
1. Robot mic captures voice (push-to-talk: user holds button)
2. PCM16 audio chunks sent to server via WebSocket (binary frames)
3. Server forwards chunks to Deepgram in real time
4. Deepgram returns partial + final transcripts
5. Robot releases button → sends {"type": "stop_recording"}
6. Server sends Deepgram Finalize → forces final transcript flush
7. Orchestrator pipeline (phases 2+3 run in parallel):
   a. Safety pre-screen (keyword check, no LLM cost)
   b. Route decision (Gemini Flash) ──┐ parallel
   c. Memory context fetch (pgvector) ─┘
   d. Sub-agent responds (Flash or Pro depending on route)
   e. Post-response: safety score, memory write, transcript store (async)
8. Response text sent to robot as JSON
9. Server creates chunked HTTP audio stream, sends URL to robot immediately
10. Robot calls audio.connecttohost(url) — starts buffering while server generates
11. ElevenLabs TTS streams MP3 chunks into queue → robot receives and plays
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
│       ├── internal.py      # Admin endpoints (X-Internal-Key)
│       └── audio.py         # GET /audio/stream/{token} — chunked MP3 for ESP32
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
│   ├── embedding_service.py # OpenAI embeddings (with in-memory cache)
│   ├── audio_stream_store.py # asyncio.Queue bridge: TTS producer → ESP32 HTTP consumer
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

### Audio Stream (public, token-gated)
```
GET    /audio/stream/{token}               — Chunked MP3 stream for ESP32 TTS playback
```

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
- Send while user is speaking (push-to-talk)
- Recommended chunk size: ~3.2 KB (100ms of audio)

**JSON control messages:**
```json
{"type": "stop_recording"}
{"type": "pong"}
{"type": "emotion_result", "emotion": "sad", "confidence": 0.85}
{"type": "music_status", "status": "playing", "genre": "calm"}
{"type": "robot_status", "battery_level": 85, "wifi_rssi": -60}
{"type": "end_session"}
```

> `stop_recording` is critical — send it when the user releases the push-to-talk button. It triggers Deepgram Finalize, which flushes the final transcript and starts the AI pipeline.

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

**TTS audio — chunked HTTP streaming:**

The server sends a URL before TTS generation starts. The robot connects to it immediately via HTTP and streams MP3 audio as chunks arrive from ElevenLabs. This approach gives ~200ms time-to-first-audio.

```json
{
  "type": "play_audio",
  "url": "https://bean-api-jolw.onrender.com/audio/stream/<token>",
  "format": "mp3",
  "turn_id": "uuid"
}
```

Robot must call:
```cpp
audio.connecttohost(url);  // ESP32-audioI2S library
```

> Do NOT wait for TTS to finish before connecting. The whole point is to connect immediately and start buffering while the server is still generating. The stream closes automatically when the response is complete.

**Music streaming (binary WebSocket frames):**
```json
{"type": "music_start", "song_id": "uuid", "title": "Calm Rain", "mood": "calm"}
```
```
<binary 8KB MP3 chunk>
<binary 8KB MP3 chunk>
...
```
```json
{"type": "music_end", "song_id": "uuid"}
{"type": "music_stopped"}
{"type": "music_volume", "volume": 70}
{"type": "music_unavailable", "mood": "calm", "message": "No calm songs available yet."}
```

> Note: Music is delivered as binary WebSocket frames (different from TTS which uses HTTP streaming). Binary frames on the WebSocket are always music.

**Keepalive:**
```json
{"type": "ping", "timestamp": "2026-03-29T05:50:47Z"}
```
Robot must respond:
```json
{"type": "pong"}
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
  |◀─ {"type":"connected"} ────────────|── Deepgram connect
  |                                    |── load user profile (cached for session)
  |                                    |
  |  [user holds button]               |
  |── [PCM16 binary chunks] ──────────▶|──▶ Deepgram STT
  |── [PCM16 binary chunks] ──────────▶|
  |◀─ {"type":"transcript_partial"} ───|◀── partial text
  |── [PCM16 binary chunks] ──────────▶|
  |  [user releases button]            |
  |── {"type":"stop_recording"} ──────▶|── Deepgram Finalize
  |◀─ {"type":"transcript_final"} ─────|◀── final text flushed
  |                                    |
  |                                    |── Orchestrator runs:
  |                                    |   ├─ routing (Gemini Flash) ──┐ parallel
  |                                    |   └─ memory fetch (pgvector) ─┘
  |                                    |   └─ sub-agent (Flash or Pro)
  |                                    |   └─ [async: safety, memory, transcript]
  |                                    |
  |◀─ {"type":"response_text"} ────────|
  |◀─ {"type":"play_audio","url":"..."} |── ElevenLabs TTS starts streaming
  |── audio.connecttohost(url) ────────▶── GET /audio/stream/<token>
  |◀─────── MP3 chunks (HTTP) ─────────|◀── chunks arrive as generated
  |── plays audio immediately          |
  |                                    |── stream ends automatically
```

### ESP32 Firmware Checklist

```
1.  Connect WebSocket: wss://bean-api-jolw.onrender.com/ws?device_id=YOUR_ID
2.  Wait for {"type": "connected"} before doing anything
3.  While user holds button: send PCM16 binary chunks from mic
4.  When user releases button: send {"type": "stop_recording"}
5.  On {"type": "play_audio"}: call audio.connecttohost(url) IMMEDIATELY
6.  On {"type": "ping"}: respond with {"type": "pong"}
7.  On binary WebSocket frames: these are music chunks — buffer and play
8.  On {"type": "music_start"}: prepare speaker for incoming binary chunks
9.  On {"type": "music_stopped"}: stop speaker
10. On {"type": "music_volume", "volume": N}: set volume 0-100
11. Send {"type": "robot_status", "battery_level": N, "wifi_rssi": N} every 60s
12. Send {"type": "emotion_result", "emotion": "...", "confidence": 0.85} when detected
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
- KeepAlive sent every 8s to prevent idle 1011 disconnect
- Finalize sent on `stop_recording` — forces final transcript flush for push-to-talk
- Auto-reconnect: up to 3 attempts with exponential backoff (2s, 4s, 8s)
- Privacy: audio exists only in RAM transit, never written to disk

### ElevenLabs Service (`services/elevenlabs_service.py`)
- Turbo v2 model (low latency)
- Streams MP3 chunks into an `asyncio.Queue` (see `audio_stream_store.py`)
- Robot receives audio via `GET /audio/stream/{token}` (chunked HTTP, not WebSocket binary)
- TTS cache: SHA-256(voice_id + text) → Supabase (7-day TTL, avoids re-generating same phrases)
- Timeouts: 10s first chunk, 8s per-chunk, 30s total

### Audio Stream Store (`services/audio_stream_store.py`)
- Bridges ElevenLabs TTS producer and ESP32 HTTP consumer via `asyncio.Queue`
- One queue per turn, keyed by a UUID token
- `create_audio_stream()` → returns `(token, queue)`
- `drain_queue(token)` → async generator yielding chunks to `StreamingResponse`
- 30s timeout on idle queue — auto-cleans on stream end or timeout
- Decouples TTS generation speed from ESP32 connection speed

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
PUBLIC_URL=https://bean-api-jolw.onrender.com
RUN_BACKGROUND_WORKERS=false
```

> `PUBLIC_URL` is required for TTS audio delivery. The server uses it to build the `play_audio` URL sent to the ESP32. Without it, the URL will point to `localhost` and the robot cannot reach the audio stream.

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
#include <ESP32-audioI2S.h>  // for audio.connecttohost()

Audio audio;

// 1. Connect WiFi
// 2. Open WebSocket
ws.connect("wss://bean-api-jolw.onrender.com/ws?device_id=ESP32_BEAN_001");

// 3. Wait for {"type":"connected"}

// 4. Push-to-talk: while button held, stream PCM16 from mic
while (button_pressed) {
    int16_t pcm[1600]; // 100ms at 16kHz mono
    mic.read(pcm, sizeof(pcm));
    ws.send_binary(pcm, sizeof(pcm));
}
// Button released — tell server to process
ws.send_text("{\"type\":\"stop_recording\"}");

// 5. On binary WebSocket frame received → music chunks (not TTS)
ws.on_binary([](uint8_t* data, size_t len) {
    speaker.play(data, len);  // music only
});

// 6. On JSON message received
ws.on_text([](String msg) {
    String type = parse_json(msg, "type");

    if (type == "play_audio") {
        // TTS response — connect to HTTP stream immediately
        String url = parse_json(msg, "url");
        audio.connecttohost(url.c_str());  // ESP32-audioI2S streams MP3
    }
    if (type == "response_text") {
        display.show(parse_json(msg, "text"));
    }
    if (type == "music_start") {
        display.show(parse_json(msg, "title"));
    }
    if (type == "music_stopped") {
        speaker.stop();
    }
    if (type == "music_volume") {
        audio.setVolume(parse_json(msg, "volume").toInt());
    }
    if (type == "ping") {
        ws.send_text("{\"type\":\"pong\"}");
    }
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

## 17. Performance Design

### Latency Breakdown (per turn)

| Stage | Typical | Notes |
|---|---|---|
| Deepgram STT | 100–300ms | Streaming, result arrives before user finishes |
| Routing + memory fetch | 400–600ms | **Run in parallel** — not sequential |
| Sub-agent (Gemini Flash) | 500–1500ms | Dominant cost — external API |
| ElevenLabs first chunk | 200–400ms | Streaming — ESP32 starts playing immediately |
| HTTP delivery | ~50ms | Chunked stream, no buffering |

### Optimizations in Place

| Optimization | Where | Saves |
|---|---|---|
| Routing + memory fetch in parallel | `orchestrator/agent.py` | ~200ms/turn |
| User profile cached at session start | `websocket_handler.py` | ~100ms/turn |
| Parallel transcript DB writes | `orchestrator/agent.py` | ~40ms/turn |
| Embedding in-memory cache (1hr TTL) | `embedding_service.py` | ~200ms on hit |
| TTS chunked HTTP — URL sent before generation | `websocket_handler.py` | ~200ms time-to-first-audio |
| Event-driven TTS/music sync | `websocket_handler.py` | Eliminates 20 poll wakeups/sec |
| Deepgram exponential backoff | `deepgram_service.py` | Faster reconnect on blips |
| Deepgram KeepAlive every 8s | `websocket_handler.py` | Prevents 1011 idle disconnect |

### Deployment Region

Deploy in **Singapore** (closest region for Southeast Asia) to reduce external API round-trip overhead by ~150ms per turn. Region cannot be changed after service creation on Render — requires redeploy.

---

## Live Deployment

- **API:** https://bean-api-jolw.onrender.com
- **WebSocket:** wss://bean-api-jolw.onrender.com/ws
- **Health:** https://bean-api-jolw.onrender.com/api/v1/health
- **Branch:** `feature/fixing`
