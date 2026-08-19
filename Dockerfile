FROM ghcr.io/astral-sh/uv:0.12.5-python3.11-trixie-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install locked runtime dependencies before copying source for better layer reuse.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --extra service --extra hf --no-install-project

COPY src ./src
COPY hf_lora/src ./hf_lora/src
COPY configs ./configs
RUN uv sync --locked --no-dev --extra service --extra hf

ENV PATH="/app/.venv/bin:$PATH" \
    MINIGPT_DEVICE=auto

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "minigpt_service.app:app", "--host", "0.0.0.0", "--port", "8000"]
