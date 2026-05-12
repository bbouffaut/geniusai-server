FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project --link-mode=copy

FROM python:3.12-slim AS runtime

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash ca-certificates libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
COPY src /app/src
COPY run.sh /app/run.sh
COPY docker/docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/run.sh /app/docker-entrypoint.sh

EXPOSE 19819

ENTRYPOINT ["/app/docker-entrypoint.sh"]
