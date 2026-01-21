FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

ENV WANDB_DIR=/wandb

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

COPY src src/

RUN uv sync --frozen

EXPOSE 8080

ENTRYPOINT ["sh", "-c", "uv run uvicorn ai_vs_human.api:app --host 0.0.0.0 --port ${PORT:-8080}"]
