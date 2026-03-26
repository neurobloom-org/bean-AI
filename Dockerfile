# ============================================================================
# BEAN AI v1 — Multi-stage Dockerfile
#
# FIX: The original builder stage used BOTH --system AND --target flags for
# `uv pip install`. These flags are mutually exclusive — uv raises an error
# when both are provided, which means `docker build` always failed.
#
# Resolution: use a Python virtual environment (/venv) in the builder stage
# so packages install into a known, self-contained path. The production and
# development stages then copy /venv from the builder and activate it.
# This keeps the final images lean and the install reproducible.
# ============================================================================

# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# libgomp1 is required for PyTorch/OpenMP (wav2vec2 emotion model)
# build-essential is needed to compile any packages with C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first so Docker layer-caches the install step
# separately from application source code changes.
COPY pyproject.toml ./
# Copy source too so hatchling can resolve the package metadata for `pip install .`
COPY . .

# Create a virtual environment and install production dependencies only.
# uv is used for speed; --no-cache-dir keeps the image smaller.
RUN pip install --no-cache-dir uv && \
    python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir uv && \
    /venv/bin/uv pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        "."

# ── Stage 2: Development ──────────────────────────────────────────────────────
FROM python:3.12-slim AS development

WORKDIR /app

# Runtime C libraries needed by PyTorch / wav2vec2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully-installed venv from the builder
COPY --from=builder /venv /venv

# Add dev extras on top of the production venv
RUN /venv/bin/pip install --no-cache-dir pytest pytest-asyncio pytest-cov ruff mypy

# Copy source code for live-reload
COPY . .

# Activate venv for all subsequent commands and runtime processes
ENV PATH="/venv/bin:$PATH"
ENV VIRTUAL_ENV=/venv
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]


# ── Stage 3: Production ───────────────────────────────────────────────────────
FROM python:3.12-slim AS production

# Non-root user for security
RUN groupadd -r bean && useradd -r -g bean -s /sbin/nologin bean

WORKDIR /app

# Runtime C libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy only the venv — no build tools in production
COPY --from=builder /venv /venv

# Copy only the application source directories (not tests, scripts, docs, etc.)
COPY agents/      ./agents/
COPY api/         ./api/
COPY background/  ./background/
COPY services/    ./services/
COPY shared/      ./shared/

# Fix ownership
RUN chown -R bean:bean /app

USER bean

# Activate venv
ENV PATH="/venv/bin:$PATH"
ENV VIRTUAL_ENV=/venv
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/v1/health')"

# Single worker — background asyncio tasks run within the same process.
# Multiple workers would each start their own background task loops,
# causing duplicate reminder checks and duplicate emotion purge runs.
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8080", \
     "--workers", "1", \
     "--log-level", "info"]
