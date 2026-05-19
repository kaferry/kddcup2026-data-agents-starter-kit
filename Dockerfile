FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --frozen --no-dev

COPY main.py ./

RUN mkdir -p /output /logs /tmp/runs

ENTRYPOINT ["/bin/sh", "-c", "uv run python main.py 2>&1 | tee /logs/runtime.log"]
