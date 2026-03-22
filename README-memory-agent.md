# Memory Agent — Custom BaseAgent

## What It Does
Parallel memory retrieval from three sources: working memory (Redis List), semantic profile (Redis JSON), and episodic memory (pgvector cosine similarity). Assembles a MemoryContext for injection into response agent prompts.

## Inputs
- `session.state["user_id"]`: Current user ID
- `session.state["current_transcript"]`: Current user transcript for episodic search

## Outputs
- `session.state["memory_context"]`: Formatted memory context string

## ADK Type
Custom `BaseAgent` with `_run_async_impl` — runs asyncio.gather for parallel lookups.

## Env Vars
- `DATABASE_URL`: PostgreSQL connection (for pgvector)
- `REDIS_URL`: Redis connection
- `OPENAI_API_KEY`: For embedding generation
