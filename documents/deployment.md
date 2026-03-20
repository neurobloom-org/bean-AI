# BEAN AI — Deployment Guide

## Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Node.js (for dev tooling only)
- A Google Cloud project with Cloud Run, Cloud SQL, and Memorystore enabled

---

## Local Development (Docker Compose)

```bash
# 1. Clone and install dev dependencies
pip install -e ".[dev]"
pre-commit install

# 2. Copy and fill in env vars
cp .env.example .env
# Edit .env — see Environment Variables section below

# 3. Start all services
docker compose up

# Services started:
#   app       → http://localhost:8080
#   postgres  → localhost:5433  (mapped from 5432 inside container)
#   redis     → localhost:6380  (mapped from 6379 inside container)

# 4. Run database migrations
docker compose exec app alembic upgrade head

# 5. Verify health
curl http://localhost:8080/api/health
```

---

## Environment Variables

All settings are loaded by `shared/config.py` via `pydantic-settings`. Every field has a default except secrets.

### Required (no defaults)

| Variable | Description |
|----------|-------------|
| `GOOGLE_API_KEY` | Gemini API key (ADK + all LlmAgents) |
| `DEEPGRAM_API_KEY` | Deepgram Nova-2 streaming STT |
| `ELEVENLABS_API_KEY` | ElevenLabs Turbo v2 TTS |
| `OPENAI_API_KEY` | text-embedding-3-small (episodic memory) |
| `TWILIO_ACCOUNT_SID` | Guardian SMS alerts |
| `TWILIO_AUTH_TOKEN` | Guardian SMS alerts |
| `TWILIO_FROM_NUMBER` | Sender number in E.164 format |
| `JWT_SECRET_KEY` | WebSocket auth token signing |

### Database & Cache

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://bean:bean@localhost:5432/bean` | PostgreSQL connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `DB_POOL_SIZE` | `10` | SQLAlchemy connection pool size |
| `REDIS_SESSION_TTL_SECONDS` | `86400` | 24-hour session TTL |
| `REDIS_TTS_CACHE_TTL_SECONDS` | `604800` | 7-day TTS cache TTL |

### Safety Thresholds

| Variable | Default | Description |
|----------|---------|-------------|
| `ALERT_THRESHOLD` | `3` | Factors needed to trigger alert (adults) |
| `MINOR_ALERT_THRESHOLD` | `2` | Factors needed to trigger alert (minors — F4 pre-set) |

### AI Models

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_FLASH_MODEL` | `gemini-2.5-flash` | Used by routing, casual chat, task, music agents |
| `GEMINI_PRO_MODEL` | `gemini-2.5-pro` | Used by therapy agent |
| `DEEPGRAM_MODEL` | `nova-2` | STT model |
| `ELEVENLABS_MODEL_ID` | `eleven_turbo_v2` | TTS model |
| `EMOTION_MODEL_NAME` | HuggingFace model ID | wav2vec2 emotion model |

### Per-Agent `.env` Files

Each agent directory has its own `.env` file for local agent-isolated testing. In production (Docker/Cloud Run), all env vars are injected at the container level from the root `.env` / Cloud Run secrets.

---

## Database Schema

8 tables created by `alembic/versions/001_initial_schema.py`:

| Table | Purpose |
|-------|---------|
| `users` | User accounts, guardian contact info, `is_minor` flag |
| `sessions` | WebSocket session tracking, start/end times |
| `conversation_turns` | Full turn history (transcript, response, route, emotion) |
| `emotion_logs` | Per-turn emotion detections (label + confidence) |
| `tasks` | Reminders and calendar tasks created via TaskAgent |
| `alert_logs` | Safety alert events (factors triggered, SMS status) |
| `music_feedback` | Music play/skip/rating events |
| `episodic_memory` | Summarised past conversations with pgvector embeddings |

### Running Migrations

```bash
# Apply all migrations
alembic upgrade head

# Check current revision
alembic current

# Create a new migration (after changing models)
alembic revision --autogenerate -m "description"

# Roll back one revision
alembic downgrade -1
```

The `episodic_memory` table requires the `pgvector` extension. The migration enables it automatically via `CREATE EXTENSION IF NOT EXISTS vector`.

---

## Background Workers

Four background services run as async tasks alongside the FastAPI app:

| Worker | File | Schedule | Purpose |
|--------|------|----------|---------|
| `reminder_check` | `background/reminder_check.py` | Every minute | Fires due reminders → TTS via active session |
| `emotion_purge` | `background/emotion_purge.py` | Daily | Deletes emotion_logs older than 90 days |
| `episodic_embedder` | `background/episodic_embedder.py` | Every 15 min | Summarises recent turns → pgvector embeddings |
| `session_cleanup` | `background/session_cleanup.py` | Hourly | Marks stale sessions as ended, purges Redis keys |

---

## Google Cloud Run Deployment

```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/bean-ai

# Deploy to Cloud Run
gcloud run deploy bean-ai \
  --image gcr.io/YOUR_PROJECT/bean-ai \
  --platform managed \
  --region us-central1 \
  --port 8080 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --set-secrets="GOOGLE_API_KEY=bean-google-api-key:latest,DEEPGRAM_API_KEY=bean-deepgram-key:latest" \
  --set-env-vars="DATABASE_URL=postgresql+asyncpg://...,REDIS_URL=redis://..."
```

**Minimum resources:** 2Gi memory (wav2vec2 model load), 2 vCPUs (concurrent WebSocket sessions).

### Cloud SQL connection

Use Cloud SQL Auth Proxy in the container or connect via private IP on VPC:

```
DATABASE_URL=postgresql+asyncpg://bean:PASSWORD@/bean?host=/cloudsql/PROJECT:REGION:INSTANCE
```

---

## Makefile Targets

```bash
make dev        # docker compose up + watch logs
make test       # pytest with coverage
make lint       # ruff check + ruff format --check
make migrate    # alembic upgrade head
make shell      # exec into running app container
make clean      # remove __pycache__, .pytest_cache, etc.
```

---

## Health Check

```
GET /api/health
```

Returns database connectivity, Redis ping, and model load status. Used by Cloud Run readiness probe and Docker Compose `healthcheck`.
