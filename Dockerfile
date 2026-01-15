FROM python:3.12-slim

# --- System dependencies ---
RUN apt-get update && apt-get install -y \
    git \
    curl \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# --- Install uv ---
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# --- Working directory ---
WORKDIR /app

# --- Copy dependency definitions first ---
COPY pyproject.toml uv.lock ./

# --- Install Python deps ---
# This creates the /app/.venv directory
RUN uv sync --frozen --no-install-project

# 🔑 CRITICAL: Add the virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# --- Copy source code ---
COPY . .

# --- Ensure runtime dirs exist ---
RUN mkdir -p models

# --- Python import path ---
ENV PYTHONPATH=/app/src

# --- Default command ---
# Recommendation: use 'python -m' to handle imports correctly
ENTRYPOINT ["python", "src/ai_vs_human/train.py"]
