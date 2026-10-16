# syntax=docker/dockerfile:1

# Base images are pinned by digest (M2 #4); both stages are Debian bookworm, so the venv
# built in the builder matches the runtime's libc. To update, pull the tag, then replace
# the digest with the one from `docker buildx imagetools inspect <tag>`.

# ---- build stage: resolve the locked environment with uv ---------------------
# Tag: ghcr.io/astral-sh/uv:python3.12-bookworm-slim
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim@sha256:e5b65587bce7de595f299855d7385fe7fca39b8a74baa261ba1b7147afa78e58 AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Install dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# Pre-download the embedding model at the revision pinned in Settings, so containers start
# without network access to Hugging Face. Only the files sentence-transformers needs.
ENV HF_HOME=/app/.hf
RUN /app/.venv/bin/python -c "\
from huggingface_hub import snapshot_download; \
from studiodesk.config import Settings; \
s = Settings(_env_file=None); \
snapshot_download(s.embedding_model, revision=s.embedding_model_revision, \
                  allow_patterns=['*.json', '*.txt', 'model.safetensors'])"

# ---- runtime stage -----------------------------------------------------------
# Tag: python:3.12-slim-bookworm
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS runtime

# All configuration comes from environment variables (Space variables/secrets); the image
# contains no .env. PORT 7860 is the Hugging Face Spaces default (`app_port` in the
# Space README); other platforms can override PORT.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    APP_ENV=prod \
    PORT=7860 \
    HF_HOME=/app/.hf \
    HF_HUB_OFFLINE=1

# Hugging Face Spaces runs containers as UID 1000, so the runtime user has UID/GID 1000.
RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
# Application files stay root-owned and are not writable by the runtime user.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.hf /app/.hf
COPY data ./data

USER 1000:1000
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"7860\")}/health', timeout=4)"

CMD ["sh", "-c", "exec uvicorn studiodesk.main:app --host 0.0.0.0 --port ${PORT}"]
