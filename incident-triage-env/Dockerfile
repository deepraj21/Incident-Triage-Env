# Serving Dockerfile for the Incident Triage environment.
# Runs the FastAPI / OpenEnv server on port 8000 (matches README app_port).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/tmp/hf-cache

WORKDIR /app

# System deps — git needed for `pip install -e` pulling openenv-core from GitHub.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

# Install deps separately from app code so changes to scenarios / tests don't
# invalidate the heavy pip layer. We copy the minimum needed for editable install.
COPY pyproject.toml README.md ./
COPY server/__init__.py ./server/
COPY models.py ./
RUN pip install -e ".[baseline]"

# Now copy the full source tree.
COPY . .

# Runtime knobs — scaling guide defaults (see docs/HF_DEPLOY.md).
ENV WORKERS=1 \
    MAX_CONCURRENT_ENVS=8 \
    PORT=8000

EXPOSE 8000

# Basic liveness probe — HF Spaces uses this to mark the Space healthy.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:${PORT:-8000}/tasks >/dev/null || exit 1

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${WORKERS:-1}"]
