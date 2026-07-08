FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev --extra ui
COPY src ./src
COPY configs ./configs
COPY ui ./ui
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ui

FROM python:3.12-slim-bookworm
WORKDIR /app
RUN groupadd --system app && useradd --system --gid app app
COPY --from=builder /app/.venv /app/.venv
COPY src ./src
COPY configs ./configs
COPY ui ./ui
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
USER app
EXPOSE 7860
CMD ["lab", "--help"]
