# ============================================================================
# BEAN AI v1 — Multi-stage Dockerfile
# ============================================================================
# This Dockerfile defines three stages:
# 1. Builder: Installs dependencies and prepares Python packages.
# 2. Development: Full environment with source code for dev iteration.
# 3. Production: Minimal, secure runtime for deployment.

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# libgomp1 is required for PyTorch/OpenMP, otherwise wav2vec2 will crash
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy all project files so the builder can see the source code directories
COPY . .

# Install 'uv' (the ultra-fast Rust resolver) to bypass pip's infinite loops
RUN pip install uv && \
    uv pip install --system --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu ".[dev]" --target /deps

# ── Stage 2: Development ──────────────────────────────────────────────────────
FROM python:3.12-slim AS development

WORKDIR /app

# Copy dependencies from the builder stage
COPY --from=builder /deps /usr/local/lib/python3.12/site-packages

# Copy all source code for development
COPY . .

# Ensure runtime has necessary C-libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Avoid Python buffering (useful for logs) and prevent writing .pyc files
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port for API server
EXPOSE 8080

# Command to start the development server with auto-reload
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]

# ── Stage 3: Production ───────────────────────────────────────────────────────
FROM python:3.12-slim AS production

# Create a non-root user for security
RUN groupadd -r bean && useradd -r -g bean bean

WORKDIR /app

# Copy only pre-built dependencies from builder
COPY --from=builder /deps /usr/local/lib/python3.12/site-packages

# Ensure runtime has necessary C-libraries
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy only required directories to reduce image size
COPY agents/    ./agents/
COPY api/       ./api/
COPY background/ ./background/
COPY services/  ./services/
COPY shared/    ./shared/

# Ensure app files are owned by non-root user
RUN chown -R bean:bean /app
USER bean

# Runtime environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

# Expose API port
EXPOSE 8080

# Healthcheck to monitor container status
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/v1/health')"

# Command to run the API server in production with a single worker
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--log-level", "info"]
