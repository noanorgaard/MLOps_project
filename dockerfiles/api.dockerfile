FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

# Set environment variables for W&B access
ENV WANDB_DIR=/wandb

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md
COPY .env .env

COPY src src/

RUN uv sync --frozen

# Expose port for FastAPI
EXPOSE 8000

ENTRYPOINT ["uv", "run", "uvicorn", "ai_vs_human.api:app", "--host", "0.0.0.0", "--port", "8000"]
