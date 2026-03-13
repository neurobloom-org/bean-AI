# BEAN AI — Backend System v4.0

BEAN is an AI-powered companion robot for teenagers (ages 13–17). Real-time voice pipeline built on Google ADK with streaming STT, emotion detection, multi-agent routing, and TTS.

## Architecture

```
Robot ←WebSocket→ BEANOrchestrator
                    ├── ParallelAgent
                    │   ├── STT (Deepgram Nova-2)
                    │   ├── Emotion (wav2vec2)
                    │   ├── Active Listen (filler phrases)
                    │   └── Alert Monitor (5-factor safety)
                    ├── MemoryAgent (Redis + pgvector)
                    ├── RoutingAgent (Gemini Flash)
                    ├── Response Agent (one of):
                    │   ├── CasualChat (Gemini Flash, 2-sentence)
                    │   ├── Therapy (Gemini Pro, SafetyService)
                    │   ├── Task (Gemini Flash + Calendar)
                    │   └── Music (Gemini Flash + WebSocket cmds)
                    └── TTS (ElevenLabs Turbo v2)
```

→ Full pipeline details in [docs/architecture.md](docs/architecture.md)

## Quick Start

```bash
git clone <repo-url>
cd bean-ai
cp .env.example .env       # fill in API keys
docker compose up -d
docker compose exec app alembic upgrade head
curl http://localhost:8080/api/health
```

→ Full env vars, Cloud Run deploy, and Makefile targets in [docs/deployment.md](docs/deployment.md)

## Development

```bash
pip install -e ".[dev]"
pre-commit install
docker compose up -d postgres redis
alembic upgrade head
uvicorn api.main:app --reload --port 8080
```

```bash
pytest tests/ -v --cov=.   # run tests
ruff check .               # lint
```

## Docs

| Document | Contents |
|----------|----------|
| [docs/architecture.md](docs/architecture.md) | Pipeline sequence, session state contract, ADK agent types, infrastructure |
| [docs/agents.md](docs/agents.md) | Every agent: model, inputs/outputs, constraints, env vars |
| [docs/deployment.md](docs/deployment.md) | Env vars, DB schema, migrations, Cloud Run, background workers |
| [docs/safety-system.md](docs/safety-system.md) | 5-factor scoring, minor thresholds, alert dispatch, limitations |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| WS | `/ws` | Robot WebSocket (audio in, audio out) |
| GET | `/api/health` | Health check |
| GET | `/api/auth/login` | Google OAuth login |
| GET | `/api/sessions/` | Session history |
| GET | `/api/sessions/{id}` | Session detail with turns |
| GET | `/api/emotions/` | Emotion logs |
| GET | `/api/tasks/` | Task list |
| PATCH | `/api/tasks/{id}` | Update task status |
| GET | `/api/alerts/` | Alert history |

## License

Proprietary — BEAN AI Project