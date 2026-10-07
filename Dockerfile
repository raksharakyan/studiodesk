# syntax=docker/dockerfile:1

# ---- build stage: resolve the locked environment with uv ---------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

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
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    APP_ENV=prod \
    PORT=8000 \
    HF_HOME=/app/.hf \
    HF_HUB_OFFLINE=1

RUN groupadd --system app && useradd --system --gid app --no-create-home app

WORKDIR /app
# Application files stay root-owned and are not writable by the runtime user.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/.hf /app/.hf
COPY data ./data

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8000\")}/health', timeout=4)"

CMD ["sh", "-c", "exec uvicorn studiodesk.main:app --host 0.0.0.0 --port ${PORT}"]
