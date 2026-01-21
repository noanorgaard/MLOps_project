FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base

COPY uv.lock uv.lock
COPY pyproject.toml pyproject.toml
COPY README.md README.md

COPY src src/

RUN uv sync --frozen

EXPOSE 8080

ENV GCS_BUCKET_NAME=ai-vs-human-monitoring
ENV REFERENCE_BLOB=reference/features.csv
ENV PREDICTION_PREFIX=prediction/

ENTRYPOINT ["sh", "-c", "uv run uvicorn ai_vs_human.data_drift_monitoring_api:app --host 0.0.0.0 --port ${PORT:-8080}"]
