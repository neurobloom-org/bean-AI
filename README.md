# BEAN AI 🤖💙

**A Privacy-First Mental Health Companion Robot for Teenagers**

BEAN is a physical, desk-based robot (ESP32-powered) that listens, detects emotion in real time, and provides therapeutic conversation grounded in CBT/DBT techniques — all while enforcing strict, database-level privacy guarantees.

> Built by a team of 3. Python 3.12 · FastAPI · Google ADK · Supabase · Docker

---

## Table of Contents

- [What BEAN Does](#what-bean-does)
- [Architecture Overview](#architecture-overview)
- [Agent System (12 Agents)](#agent-system-12-agents)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Environment Variables](#environment-variables)
- [Database](#database)
- [API Reference](#api-reference)
- [WebSocket Protocol](#websocket-protocol)
- [Privacy Guarantees](#privacy-guarantees)
- [Safety System](#safety-system)
- [Team & Branch Guide](#team--branch-guide)
- [Contributing](#contributing)

---

## What BEAN Does

A teenager talks to BEAN. The robot:

1. **Listens** — Streams PCM16 audio from the ESP32 microphone to Deepgram (STT)
2. **Feels** — Runs wav2vec2 emotion detection in parallel (8 labels: angry, calm, fearful, happy, neutral, sad, surprised, disgusted)
3. **Thinks** — Orchestrator routes to the right agent: casual chat, therapy, task management, music, or crisis alert
4. **Responds** — Generates a reply via Gemini and streams audio back via ElevenLabs (TTS)
5. **Remembers** — Stores episodic memory as vector embeddings. Raw transcripts are hard-deleted after 24 hours.
6. **Protects** — 5-factor safety screening runs in parallel every turn. SMS alert fires to linked guardian on crisis detection.

---

## Architecture Overview

```
ESP32 (Hardware)
    │  PCM16 audio stream (WebSocket)
    ▼
WebSocket Handler  ──►  STT Agent (Deepgram)
                   ──►  Emotion Agent (wav2vec2)  ──►  Emotion Logs (Supabase)
                   ──►  Orchestrator Agent
                            │
                    ┌───────┼───────────────────────┐
                    ▼       ▼                       ▼
              Routing    Memory Agent          Safety Service
              Agent      (pgvector search)    (5-factor screen)
                    │
          ┌─────────┼──────────────────────────────┐
          ▼         ▼              ▼                ▼
    CasualChat  TherapyAgent   TaskAgent       MusicAgent
    (Flash)     (Pro + RAG)    (Flash +        (Flash +
                               Calendar)       ESP32 cmd)
                    │                               │
                    ▼                               ▼
             AlertAgent                     Active Listen Agent
             (Twilio SMS)                   (Filler phrases)
                    │
                    ▼
             TTS Agent (ElevenLabs) ──► ESP32
```

**LLM Strategy — Two tiers:**

| Tier | Model | Used For |
|------|-------|----------|
| Flash (cheap) | `gemini-2.0-flash` | Routing, casual chat, task, music, filler |
| Pro (quality) | `gemini-2.5-pro` | Therapy, crisis response, full safety assessment |

---

## Agent System (12 Agents)

| Agent | Owner | Role |
|-------|-------|------|
| `orchestrator` | Dev B | Central coordinator — runs all 5 phases per turn |
| `routing` | Person 1 | Classifies message into casual / therapy / task / music / alert |
| `memory` | Dev B | Fetches user profile + episodic memories via pgvector |
| `memory_writer` | Dev B | Writes new episodic memories post-turn (non-blocking) |
| `casual_chat` | Dev B | Short, warm persona responses (Flash, 2-sentence target) |
| `therapeutic_convo` | Person 2 | CBT/DBT therapeutic conversation (Pro + RAG) |
| `task` | Dev B | Reminder & Google Calendar management (Flash) |
| `music` | Dev B | Music playback commands to ESP32 (Flash) |
| `alert` | Dev B | Crisis detection & Twilio SMS to guardian |
| `emotion` | Dev B | wav2vec2 emotion classification (fire-and-forget) |
| `stt` | Dev B | Deepgram Nova-2 streaming transcription |
| `tts` | Dev B | ElevenLabs Turbo v2 audio synthesis & streaming |
| `active_listen` | Dev B | Filler phrase delivery while response generates |

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Framework | FastAPI 0.115+, Python 3.12 |
| Agent Framework | Google ADK (Agent Development Kit) |
| LLMs | Gemini 2.0 Flash · Gemini 2.5 Pro |
| Database | Supabase (PostgreSQL + pgvector + pg_cron) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| STT | Deepgram Nova-2 (streaming) |
| TTS | ElevenLabs Turbo v2 (streaming) |
| Emotion | HuggingFace `wav2vec2` (local inference) |
| Alerts | Twilio SMS |
| Calendar | Google Calendar API (OAuth 2.0) |
| Realtime | WebSockets (bidirectional ESP32 ↔ backend) |
| Caching | Redis |
| Containerisation | Docker + Docker Compose |

---

## Project Structure

```
bean-ai/
├── agents/                   # 12 ADK agents
│   ├── orchestrator/         # Central coordinator
│   ├── routing/              # Message classifier (Person 1)
│   ├── therapeutic_convo/    # CBT/DBT therapy (Person 2)
│   ├── casual_chat/
│   ├── task/
│   ├── music/
│   ├── alert/
│   ├── memory/
│   ├── memory_writer/
│   ├── emotion/
│   ├── stt/
│   ├── tts/
│   └── active_listen/
├── api/
│   ├── main.py               # FastAPI app entrypoint + lifespan
│   ├── websocket_handler.py  # ESP32 real-time audio pipeline
│   ├── middleware/
│   │   ├── auth_middleware.py
│   │   └── rate_limiter.py
│   └── routes/
│       ├── auth.py           # Google OAuth flow
│       ├── sessions.py
│       ├── alerts.py
│       ├── emotions.py
│       ├── tasks.py
│       ├── guardian.py       # Guardian dashboard endpoints
│       └── health.py
├── background/               # Async background loops
│   ├── session_cleanup.py    # Transcript purge + session expiry
│   ├── episodic_embedder.py  # Async memory vectorisation
│   ├── emotion_purge.py
│   └── reminder_check.py
├── services/                 # Third-party API wrappers
│   ├── supabase_client.py
│   ├── llm_service.py
│   ├── deepgram_service.py
│   ├── elevenlabs_service.py
│   ├── embedding_service.py
│   ├── rag_service.py        # pgvector CBT/DBT retrieval
│   ├── safety_service.py     # 5-factor crisis screening
│   ├── privacy_service.py    # Transcript lifecycle management
│   ├── calendar_service.py
│   └── twilio_service.py
├── shared/                   # Types imported by every module
│   ├── config.py             # Pydantic Settings singleton
│   ├── enums.py              # RouteType, AlertLevel, EmotionLabel
│   ├── schemas.py            # All Pydantic v2 models
│   └── exceptions.py        # BEANError hierarchy
├── supabase/
│   ├── config.toml
│   └── migrations/
│       ├── 001_schema.sql    # 12 tables + pgvector
│       ├── 002_rls_policies.sql
│       ├── 003_cron_jobs.sql
│       ├── 004_functions.sql # pgvector RPC + emotion summary
│       └── 005_rag_seed.sql  # ~20 CBT/DBT techniques
├── tests/                    # pytest test suite
├── scripts/
│   └── seed_rag_embeddings.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── Makefile
└── .env.example
```

---

## Getting Started

### Prerequisites

- Docker Desktop
- Git
- API keys for: Supabase, Google (Gemini + OAuth), Deepgram, ElevenLabs, OpenAI, Twilio

### 1. Clone and configure

```bash
git clone https://github.com/your-org/bean-ai.git
cd bean-ai
cp .env.example .env
# Fill in all values in .env
```

### 2. Install PyTorch (CPU-only — saves ~4GB)

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 3. Start the server

```bash
make dev
# FastAPI runs at http://localhost:8080
# Hot-reload enabled
```

### 4. Run migrations

```bash
make migrate        # Push schema to Supabase cloud
make seed           # Populate RAG embeddings
```

### 5. Verify health

```
GET http://localhost:8080/api/v1/health
```

---

## Environment Variables

Copy `.env.example` → `.env`. Never commit `.env`.

Key variables:

| Variable | Where to get it |
|----------|----------------|
| `SUPABASE_URL` | Supabase Dashboard → Project Settings → API |
| `SUPABASE_SERVICE_ROLE_KEY` | Same as above |
| `GOOGLE_API_KEY` | [aistudio.google.com](https://aistudio.google.com) |
| `DEEPGRAM_API_KEY` | [deepgram.com](https://deepgram.com) |
| `ELEVENLABS_API_KEY` + `ELEVENLABS_VOICE_ID` | ElevenLabs dashboard (paid plan required) |
| `OPENAI_API_KEY` | For embeddings only |
| `TWILIO_ACCOUNT_SID` / `TWILIO_AUTH_TOKEN` | Twilio console |

See `.env.example` for the full list including privacy retention settings and rate limits.

---

## Database

State lives entirely in **Supabase** (no local DB container).

### Tables (12 total)

| Table | Purpose |
|-------|---------|
| `user_profiles` | User facts, preferences (JSON) |
| `guardian_links` | Patient ↔ guardian relationships + access flags |
| `sessions` | Active/past conversation sessions |
| `session_transcripts` | 24h rolling transcript window |
| `emotion_logs` | Per-turn emotion data (90-day retention) |
| `episodic_memories` | Long-term memories as `vector(1536)` |
| `rag_techniques` | CBT/DBT technique library as `vector(1536)` |
| `alerts` | Crisis alerts with severity levels |
| `tasks` | User reminders & calendar tasks |
| `oauth_tokens` | Encrypted Google OAuth tokens |
| `rate_limits` | Per-user API rate tracking |
| `tts_cache` | Pre-cached TTS audio for filler phrases |

### Scheduled jobs (pg_cron)

| Job | Schedule |
|-----|----------|
| Purge transcripts older than 24h | Hourly |
| Purge old emotion logs | Daily 3am UTC |
| Expire inactive sessions | Every 30 min |
| Clean stale rate limit records | Hourly |

### Stored functions

- `search_episodic_memories(user_uuid, query_embedding, threshold, count)` — pgvector cosine search
- `get_patient_emotion_summary(patient_uuid, days_back)` — JSONB emotion aggregation for guardian dashboard
- `upsert_user_profile_facts(user_uuid, facts JSONB)` — atomic array merge for memory writer

---

## API Reference

Base URL: `http://localhost:8080/api/v1`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check (DB + Redis status) |
| POST | `/auth/google` | Initiate Google OAuth |
| GET | `/auth/google/callback` | OAuth callback |
| POST | `/sessions` | Create conversation session |
| GET | `/sessions/{id}` | Get session details |
| GET | `/alerts` | List alerts for user |
| GET | `/emotions` | Emotion history (aggregated) |
| POST | `/tasks` | Create task / reminder |
| GET | `/tasks` | List tasks |
| GET | `/guardian/patients` | Guardian: list linked patients |
| GET | `/guardian/patients/{id}/emotions` | Guardian: patient emotion trends |

All routes require `Authorization: Bearer <supabase_jwt>` except `/health` and `/auth/*`.

---

## WebSocket Protocol

Connect: `ws://localhost:8080/ws`
Auth: Pass JWT via `Sec-WebSocket-Protocol` header (not query param).

### Client → Server (ESP32 sends)

| Message | Format | Description |
|---------|--------|-------------|
| Audio frame | Binary (PCM16 bytes) | Raw microphone audio |
| `{"type": "utterance_end"}` | JSON | Silence detected, process transcript |
| `{"type": "ping"}` | JSON | Keepalive |

### Server → Client (ESP32 receives)

| Message | Description |
|---------|-------------|
| `{"type": "transcript_result", "text": "..."}` | STT result |
| `{"type": "filler_audio", "audio": "<b64>"}` | Immediate filler while thinking |
| `{"type": "response_text", "text": "..."}` | BEAN's response text |
| `{"type": "response_audio", "audio": "<b64>"}` | TTS audio chunk |
| `{"type": "music_command", "command": "play", ...}` | Music playback instruction |
| `{"type": "alert", "level": "CRISIS"}` | Safety alert fired |
| `{"type": "error", "message": "..."}` | Error notification |

---

## Privacy Guarantees

| Guarantee | How |
|-----------|-----|
| No audio storage | Voice bytes processed in RAM only — never written to disk or DB |
| 24h transcript expiry | Hard-deleted by background loop + pg_cron (dual enforcement) |
| 90-day emotion retention | Aggregated data only after purge |
| Guardian isolation | RLS policies at DB level — guardians cannot query conversation text |
| No secrets in images | `.env` excluded from Docker build via `.dockerignore` |
| Non-root container | Production image runs as `bean` user |

---

## Safety System

Every conversation turn runs a **5-factor safety assessment**:

| Factor | What It Checks |
|--------|---------------|
| F1 | Crisis keywords (suicide, self-harm, abuse) |
| F2 | Explicit statements of intent |
| F3 | Escalating distress across the session |
| F4 | Contextual risk (isolation, hopelessness language) |
| F5 | Guardian-flagged high-risk status |

**Alert levels:** `NONE → LOW → MEDIUM → HIGH → CRISIS`

On `CRISIS`: Twilio SMS fires to the linked guardian immediately. 5-minute cooldown between alerts per user.

---

## Team & Branch Guide

| Branch | Owner | Purpose |
|--------|-------|---------|
| `feature/project-init` | Anyone | Docker, env, Makefile, README |
| `feature/shared-foundation` | Dev A | `shared/` — enums, schemas, config, exceptions |
| `feature/database-setup` | Dev A | All 5 SQL migrations + `supabase/config.toml` |
| `feature/ci-pipeline` | Dev B | GitHub Actions, PR template, CODEOWNERS |
| `feature/external-services` | Dev C | All `services/` wrappers |
| `feature/routing-agent` | Person 1 | `agents/routing/` |
| `feature/therapy-agent` | Person 2 | `agents/therapeutic_convo/` + RAG |
| `feature/orchestrator` | Dev B | `agents/orchestrator/` |
| `feature/api-routes` | Dev B | `api/routes/` + middleware |
| `feature/websocket` | Dev B | `api/websocket_handler.py` |
| `feature/background-tasks` | Dev B | `background/` loops |
| `feature/tests` | Dev C | Full test suite + seed script |

**Merge order:** `project-init` → `shared-foundation` → `database-setup` → everything else in parallel.

All branches target `develop`. Only `develop` → `main` after full integration testing.

---

## Makefile Commands

```bash
make dev          # Start with hot-reload (Docker)
make build        # Rebuild Docker image
make test         # Run pytest with coverage
make test-fast    # pytest, no coverage, stop on first fail
make lint         # ruff check
make format       # ruff format
make typecheck    # mypy strict
make migrate      # Push migrations to Supabase cloud
make seed         # Populate RAG embeddings
make logs         # Tail API logs
make shell        # Shell into running container
make clean        # Remove __pycache__, .pytest_cache etc.
```

---

## Contributing

1. Branch off `develop`: `git checkout -b feature/your-feature`
2. Run `make lint`, `make typecheck`, `make test` before pushing
3. If you add an env var → update `.env.example` and tell the team
4. If you add a dependency → update `pyproject.toml` and tell the team
5. PRs target `develop` only — never `main`
6. See `CONTRIBUTING.md` and `bean_ai_branch_guide.docx` for the full protocol

---

*BEAN AI — Built with care for the teenagers who need it most.*
