.PHONY: dev build up down logs shell test lint format migrate seed clean install

# ── Development ───────────────────────────────────────────────────────────────
# Install Python dependencies including dev tools
install:
	@echo "Installing dependencies (forcing PyTorch CPU wheels)..."
	pip install --extra-index-url https://download.pytorch.org/whl/cpu -e '.[dev]'

# Start development environment in foreground using Docker Compose
dev:
	docker compose up --build

# Build Docker images without starting containers
build:
	docker compose build

# Start Docker Compose in detached mode
up:
	docker compose up -d

# Stop Docker Compose and remove containers
down:
	docker compose down

# Stream logs from the 'bean-api' container
logs:
	docker compose logs -f bean-api

# Open a shell inside the 'bean-api' container
shell:
	docker compose exec bean-api /bin/bash

# ── Testing ───────────────────────────────────────────────────────────────────
# Run all tests with coverage reporting
test:
	pytest tests/ -v --cov=. --cov-report=term-missing

# Run tests quickly, stopping at the first failure, without coverage
test-fast:
	pytest tests/ -v -x --no-cov

# ── Code Quality ──────────────────────────────────────────────────────────────
# Lint the code using ruff
lint:
	ruff check .

# Automatically format code using ruff
format:
	ruff format .

# Type-check Python code using mypy
typecheck:
	mypy .

# ── Database ──────────────────────────────────────────────────────────────────
# Apply database migrations to Supabase
migrate:
	supabase db push

# Reset local Supabase database (destructive)
migrate-local:
	supabase db reset

# ── Setup ─────────────────────────────────────────────────────────────────────
# Seed the database with initial data for RAG embeddings
seed:
	python scripts/seed_rag_embeddings.py

# ── Cleanup ───────────────────────────────────────────────────────────────────
# Remove Python cache files, compiled files, and temporary test/lint caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
